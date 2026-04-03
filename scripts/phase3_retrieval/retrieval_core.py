from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from db_utils import (
    connect_db,
    create_retrieval_run,
    fetch_experiment_rows,
    fetch_feature_type_map,
    get_or_create_query,
    load_gallery_feature_matrix,
    load_gallery_rows,
    parse_vector_json,
    replace_retrieval_results,
    upsert_query_features,
)
from feature_utils import (
    FEATURE_SPECS,
    REGIONAL_GRID,
    center_square_crop_resize,
    chi_square_similarity_scores,
    cosine_similarity_scores,
    euclidean_similarity_scores,
    extract_all_features,
    load_cnn_model,
    select_device,
)


def _compute_similarity(metric_name: str, query_vector: np.ndarray, gallery_matrix: np.ndarray) -> np.ndarray:
    if metric_name == "cosine_similarity":
        return cosine_similarity_scores(query_vector, gallery_matrix)
    if metric_name == "chi_square_distance_then_inverse":
        return chi_square_similarity_scores(query_vector, gallery_matrix)
    if metric_name == "euclidean_distance_then_inverse":
        return euclidean_similarity_scores(query_vector, gallery_matrix)
    raise ValueError(f"Unsupported similarity metric: {metric_name}")


class RetrievalEngine:
    def __init__(
        self,
        db_path: Path,
        processed_root: Path,
        device: str = "auto",
    ) -> None:
        self.db_path = Path(db_path).resolve()
        self.processed_root = Path(processed_root).resolve()
        self.connection = connect_db(self.db_path)
        self.feature_type_map = fetch_feature_type_map(self.connection)
        self.experiment_rows = fetch_experiment_rows(self.connection)
        self.gallery_rows = load_gallery_rows(self.connection, keep_only=True)
        self.gallery_by_image_id = {int(row["image_id"]): row for row in self.gallery_rows}
        self.gallery_image_ids = [int(row["image_id"]) for row in self.gallery_rows]
        self.feature_matrices = {
            feature_name: load_gallery_feature_matrix(
                self.connection,
                self.gallery_image_ids,
                feature_type_id,
            )
            for feature_name, feature_type_id in self.feature_type_map.items()
        }
        self.device = select_device(device)
        self._cnn_model = None

    def close(self) -> None:
        self.connection.close()

    def _ensure_cnn_model(self):
        if self._cnn_model is None:
            self._cnn_model = load_cnn_model("resnet18", self.device)
        return self._cnn_model

    def _load_experiment(self, experiment_name: str) -> dict:
        row = self.experiment_rows.get(experiment_name)
        if row is None:
            raise ValueError(f"Experiment '{experiment_name}' is not registered in the database.")
        return {
            "experiment_id": int(row["experiment_id"]),
            "name": str(row["name"]),
            "weights": json.loads(str(row["weighting_json"])),
            "notes": str(row["notes"] or ""),
        }

    def _load_gallery_query_features(self, image_id: int, required_features: List[str]) -> Dict[str, np.ndarray]:
        if image_id not in self.gallery_by_image_id:
            raise ValueError(f"image_id {image_id} was not found in the gallery database.")
        index = self.gallery_image_ids.index(int(image_id))
        return {
            feature_name: self.feature_matrices[feature_name][index]
            for feature_name in required_features
        }

    def _prepare_external_query_features(self, image_path: Path, required_features: List[str]) -> Dict[str, np.ndarray]:
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        use_cnn = "cnn_embedding" in required_features
        with Image.open(image_path) as image:
            normalized = center_square_crop_resize(image)
            return extract_all_features(
                image=normalized,
                bins=(8, 8, 8),
                regional_grid=REGIONAL_GRID,
                cnn_model=self._ensure_cnn_model() if use_cnn else None,
                device=self.device if use_cnn else None,
            )

    def _build_query_record(
        self,
        query_image_id: int | None,
        query_image_path: str | None,
    ) -> tuple[int, str, str, dict]:
        if query_image_id is not None:
            gallery_row = self.gallery_by_image_id[int(query_image_id)]
            preprocess_row = self.connection.execute(
                """
                SELECT preprocess_params_json
                FROM preprocessing_metadata
                WHERE image_id = ?
                """,
                (int(query_image_id),),
            ).fetchone()
            preprocess_params = json.loads(str(preprocess_row["preprocess_params_json"])) if preprocess_row else {}
            relative_path = str(gallery_row["processed_relative_path"])
            query_id = get_or_create_query(
                self.connection,
                query_source_type="gallery_image",
                query_image_path=relative_path,
                preprocess_params_json=json.dumps(preprocess_params, ensure_ascii=True),
            )
            return query_id, "gallery_image", relative_path, preprocess_params

        absolute_path = str(Path(query_image_path or "").resolve())
        preprocess_params = {
            "method": "center_square_crop_resize",
            "target_size": [224, 224],
        }
        query_id = get_or_create_query(
            self.connection,
            query_source_type="external_image",
            query_image_path=absolute_path,
            preprocess_params_json=json.dumps(preprocess_params, ensure_ascii=True),
        )
        return query_id, "external_image", absolute_path, preprocess_params

    def run_retrieval(
        self,
        experiment_name: str,
        top_k: int = 5,
        query_image_id: int | None = None,
        query_image_path: str | None = None,
        persist: bool = True,
    ) -> dict:
        if query_image_id is None and not query_image_path:
            raise ValueError("Provide query_image_id or query_image_path.")

        experiment = self._load_experiment(experiment_name)
        required_features = sorted(experiment["weights"].keys())

        if query_image_id is not None:
            query_vectors = self._load_gallery_query_features(query_image_id, required_features)
        else:
            query_vectors = self._prepare_external_query_features(Path(str(query_image_path)), required_features)

        per_feature_scores: Dict[str, np.ndarray] = {}
        fused_scores = np.zeros((len(self.gallery_rows),), dtype=np.float32)
        for feature_name in required_features:
            metric = FEATURE_SPECS[feature_name]["similarity_metric"]
            score_vector = _compute_similarity(metric, query_vectors[feature_name], self.feature_matrices[feature_name])
            per_feature_scores[feature_name] = score_vector
            fused_scores += float(experiment["weights"][feature_name]) * score_vector

        if query_image_id is not None:
            exclude_index = self.gallery_image_ids.index(int(query_image_id))
            fused_scores[exclude_index] = -np.inf

        ranked_indices = np.argsort(-fused_scores)
        ranked_rows = []
        for ranked_index in ranked_indices[:top_k]:
            gallery_row = self.gallery_rows[int(ranked_index)]
            ranked_rows.append(
                {
                    "rank": len(ranked_rows) + 1,
                    "image_id": int(gallery_row["image_id"]),
                    "species_id": int(gallery_row["species_id"]) if gallery_row["species_id"] is not None else None,
                    "species_name": str(gallery_row["species_name"] or ""),
                    "processed_relative_path": str(gallery_row["processed_relative_path"]),
                    "per_feature_scores": {
                        feature_name: float(per_feature_scores[feature_name][ranked_index])
                        for feature_name in required_features
                    },
                    "fused_score": float(fused_scores[ranked_index]),
                }
            )

        query_id = None
        run_id = None
        if persist:
            query_id, _, _, _ = self._build_query_record(query_image_id, query_image_path)
            upsert_query_features(self.connection, query_id, self.feature_type_map, query_vectors)
            run_id = create_retrieval_run(
                self.connection,
                query_id=query_id,
                experiment_id=int(experiment["experiment_id"]),
                mode=experiment_name,
                top_k=top_k,
            )
            replace_retrieval_results(self.connection, run_id, ranked_rows)

        return {
            "query_id": query_id,
            "run_id": run_id,
            "experiment_name": experiment_name,
            "top_k": top_k,
            "results": ranked_rows,
        }

    def load_query_features_from_db(self, query_id: int) -> Dict[str, np.ndarray]:
        rows = self.connection.execute(
            """
            SELECT ft.name AS feature_name, qf.vector_json
            FROM query_features AS qf
            JOIN feature_types AS ft ON ft.feature_type_id = qf.feature_type_id
            WHERE qf.query_id = ?
            ORDER BY ft.feature_type_id
            """,
            (int(query_id),),
        ).fetchall()
        return {str(row["feature_name"]): parse_vector_json(str(row["vector_json"])) for row in rows}
