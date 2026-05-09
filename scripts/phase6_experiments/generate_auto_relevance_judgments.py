from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from db_utils import connect_db, fetch_feature_type_map, load_gallery_feature_matrix
from feature_utils import (
    chi_square_similarity_scores,
    cosine_similarity_scores,
    euclidean_similarity_scores,
    load_csv_rows,
    write_csv_rows,
)


AUTO_VISUAL_WEIGHTS = {
    "regional_hsv_hist": 0.25,
    "global_hsv_hist": 0.15,
    "color_moments": 0.15,
    "lbp_hist": 0.15,
    "hog_descriptor": 0.15,
    "silhouette_shape_descriptor": 0.10,
    "cnn_embedding": 0.05,
}

SIMILARITY_METRICS = {
    "global_hsv_hist": "chi_square",
    "regional_hsv_hist": "chi_square",
    "color_moments": "euclidean",
    "lbp_hist": "chi_square",
    "hog_descriptor": "cosine",
    "silhouette_shape_descriptor": "euclidean",
    "cnn_embedding": "cosine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate transparent auto-assisted visual relevance labels from descriptor similarity. "
            "This is not a replacement for human manual relevance."
        )
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument(
        "--template-csv",
        default="outputs/manual_relevance/manual_relevance_template.csv",
        help="Manual relevance template generated from latest retrieval results.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/auto_relevance/auto_visual_relevance.csv",
        help="CSV with relevance_grade prefilled from the auto visual proxy.",
    )
    parser.add_argument(
        "--absolute-thresholds",
        default="0.54,0.62,0.70",
        help="Comma-separated grade 1/2/3 lower bounds before per-query quantile adjustment.",
    )
    return parser.parse_args()


def _compute_pair_similarity(feature_name: str, query_vector: np.ndarray, candidate_vector: np.ndarray) -> float:
    metric = SIMILARITY_METRICS[feature_name]
    matrix = candidate_vector.reshape(1, -1)
    if metric == "cosine":
        return float(cosine_similarity_scores(query_vector, matrix)[0])
    if metric == "chi_square":
        return float(chi_square_similarity_scores(query_vector, matrix)[0])
    if metric == "euclidean":
        return float(euclidean_similarity_scores(query_vector, matrix)[0])
    raise ValueError(f"Unsupported metric for {feature_name}: {metric}")


def _grade_scores_by_query(rows: list[dict], absolute_thresholds: tuple[float, float, float]) -> None:
    by_query: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[int(row["query_id"])].append(row)

    for query_rows in by_query.values():
        scores = np.asarray([float(row["auto_visual_score"]) for row in query_rows], dtype=np.float32)
        if scores.size == 0:
            continue
        q40, q65, q85 = np.quantile(scores, [0.40, 0.65, 0.85])
        grade_1 = max(float(absolute_thresholds[0]), float(q40))
        grade_2 = max(float(absolute_thresholds[1]), float(q65))
        grade_3 = max(float(absolute_thresholds[2]), float(q85))

        for row in query_rows:
            score = float(row["auto_visual_score"])
            if score >= grade_3:
                grade = 3
            elif score >= grade_2:
                grade = 2
            elif score >= grade_1:
                grade = 1
            else:
                grade = 0
            row["relevance_grade"] = grade
            row["review_notes"] = (
                "auto_visual_proxy: descriptor-only grade, not human manual relevance; "
                f"thresholds q/abs=({grade_1:.4f},{grade_2:.4f},{grade_3:.4f})"
            )


def main() -> int:
    args = parse_args()
    template_rows = load_csv_rows(Path(args.template_csv).resolve())
    if not template_rows:
        raise ValueError("Manual relevance template is empty. Run prepare_manual_relevance.py first.")

    thresholds = tuple(float(item.strip()) for item in str(args.absolute_thresholds).split(",") if item.strip())
    if len(thresholds) != 3:
        raise ValueError("--absolute-thresholds must contain exactly three comma-separated numbers.")

    image_ids = sorted(
        {
            int(row["query_image_id"])
            for row in template_rows
            if str(row.get("query_image_id", "")).strip()
        }
        | {int(row["candidate_image_id"]) for row in template_rows}
    )
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}

    connection = connect_db(Path(args.db_path).resolve())
    try:
        feature_type_map = fetch_feature_type_map(connection)
        matrices = {
            feature_name: load_gallery_feature_matrix(connection, image_ids, feature_type_map[feature_name])
            for feature_name in AUTO_VISUAL_WEIGHTS
        }
    finally:
        connection.close()

    export_rows = []
    for row in template_rows:
        query_image_id_text = str(row.get("query_image_id", "")).strip()
        if not query_image_id_text:
            continue

        query_index = image_index[int(query_image_id_text)]
        candidate_index = image_index[int(row["candidate_image_id"])]
        score = 0.0
        for feature_name, weight in AUTO_VISUAL_WEIGHTS.items():
            score += float(weight) * _compute_pair_similarity(
                feature_name,
                matrices[feature_name][query_index],
                matrices[feature_name][candidate_index],
            )

        export_row = dict(row)
        export_row["auto_visual_score"] = f"{score:.6f}"
        export_rows.append(export_row)

    _grade_scores_by_query(export_rows, thresholds)  # type: ignore[arg-type]

    output_csv = Path(args.output_csv).resolve()
    write_csv_rows(
        output_csv,
        export_rows,
        [
            "query_id",
            "query_image_id",
            "query_species_name",
            "query_processed_relative_path",
            "candidate_image_id",
            "candidate_species_name",
            "candidate_processed_relative_path",
            "source_experiments",
            "best_rank_seen",
            "auto_visual_score",
            "relevance_grade",
            "review_notes",
        ],
    )

    grade_counts = defaultdict(int)
    for row in export_rows:
        grade_counts[int(row["relevance_grade"])] += 1
    summary = ", ".join(f"{grade}:{grade_counts[grade]}" for grade in sorted(grade_counts))
    print(f"Saved auto visual relevance CSV: {output_csv}")
    print(f"Auto visual grade counts: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
