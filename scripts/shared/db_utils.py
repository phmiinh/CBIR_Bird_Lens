from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from feature_utils import DEFAULT_EXPERIMENTS

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    source_relative_path TEXT NOT NULL,
    processed_relative_path TEXT NOT NULL,
    species_id INTEGER,
    species_name TEXT,
    split TEXT,
    width INTEGER,
    height INTEGER,
    is_perching INTEGER,
    is_side_view INTEGER,
    keep INTEGER
);

CREATE TABLE IF NOT EXISTS preprocessing_metadata (
    image_id INTEGER PRIMARY KEY,
    bbox_x REAL,
    bbox_y REAL,
    bbox_w REAL,
    bbox_h REAL,
    crop_left INTEGER,
    crop_top INTEGER,
    crop_right INTEGER,
    crop_bottom INTEGER,
    target_width INTEGER,
    target_height INTEGER,
    padding_ratio REAL,
    keep INTEGER,
    notes TEXT,
    preprocess_params_json TEXT,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feature_types (
    feature_type_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    vector_dim INTEGER NOT NULL,
    similarity_metric TEXT NOT NULL,
    extraction_params_json TEXT NOT NULL,
    is_primary INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS image_features (
    image_id INTEGER NOT NULL,
    feature_type_id INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY (image_id, feature_type_id),
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE,
    FOREIGN KEY (feature_type_id) REFERENCES feature_types(feature_type_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS queries (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_source_type TEXT NOT NULL,
    query_image_path TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    preprocess_params_json TEXT NOT NULL,
    UNIQUE (query_source_type, query_image_path)
);

CREATE TABLE IF NOT EXISTS query_features (
    query_id INTEGER NOT NULL,
    feature_type_id INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY (query_id, feature_type_id),
    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
    FOREIGN KEY (feature_type_id) REFERENCES feature_types(feature_type_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    feature_set_json TEXT NOT NULL,
    weighting_json TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    summary_metrics_json TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    mode TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_results (
    run_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    per_feature_scores_json TEXT NOT NULL,
    fused_score REAL NOT NULL,
    PRIMARY KEY (run_id, rank),
    FOREIGN KEY (run_id) REFERENCES retrieval_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    query_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    judgment_source TEXT NOT NULL,
    relevance_grade INTEGER NOT NULL,
    PRIMARY KEY (query_id, image_id, judgment_source),
    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE CASCADE,
    FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_images_species_id ON images(species_id);
CREATE INDEX IF NOT EXISTS idx_image_features_feature_type_id ON image_features(feature_type_id);
CREATE INDEX IF NOT EXISTS idx_query_features_feature_type_id ON query_features(feature_type_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_query_id ON retrieval_runs(query_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_experiment_id ON retrieval_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_runs_latest ON retrieval_runs(query_id, experiment_id, run_id DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_results_image_id ON retrieval_results(image_id);
CREATE INDEX IF NOT EXISTS idx_relevance_judgments_source ON relevance_judgments(judgment_source);
CREATE INDEX IF NOT EXISTS idx_relevance_judgments_source_query ON relevance_judgments(judgment_source, query_id, image_id);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def vector_to_json(vector: Sequence[float] | np.ndarray, decimals: int = 6) -> str:
    rounded = np.round(np.asarray(vector, dtype=np.float32), decimals=decimals)
    return json_dumps([float(value) for value in rounded.tolist()])


def parse_vector_json(vector_json: str) -> np.ndarray:
    return np.asarray(json.loads(vector_json), dtype=np.float32)


def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)
    connection.commit()


def insert_images(connection: sqlite3.Connection, rows: Iterable[dict]) -> None:
    values = [
        (
            int(row["image_id"]),
            row["source_relative_path"],
            row["processed_relative_path"],
            int(row["species_id"]) if str(row.get("species_id", "")).strip() else None,
            row.get("species_name", ""),
            row.get("split", ""),
            int(row["width"]) if str(row.get("width", "")).strip() else None,
            int(row["height"]) if str(row.get("height", "")).strip() else None,
            int(row["is_perching"]) if str(row.get("is_perching", "")).strip() else None,
            int(row["is_side_view"]) if str(row.get("is_side_view", "")).strip() else None,
            int(row["keep"]) if str(row.get("keep", "")).strip() else None,
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO images (
            image_id, source_relative_path, processed_relative_path, species_id, species_name,
            split, width, height, is_perching, is_side_view, keep
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def insert_preprocessing_metadata(connection: sqlite3.Connection, rows: Iterable[dict]) -> None:
    values = [
        (
            int(row["image_id"]),
            row.get("bbox_x"),
            row.get("bbox_y"),
            row.get("bbox_w"),
            row.get("bbox_h"),
            row.get("crop_left"),
            row.get("crop_top"),
            row.get("crop_right"),
            row.get("crop_bottom"),
            row.get("target_width"),
            row.get("target_height"),
            row.get("padding_ratio"),
            row.get("keep"),
            row.get("notes", ""),
            row.get("preprocess_params_json", "{}"),
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO preprocessing_metadata (
            image_id, bbox_x, bbox_y, bbox_w, bbox_h, crop_left, crop_top, crop_right, crop_bottom,
            target_width, target_height, padding_ratio, keep, notes, preprocess_params_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def insert_feature_types(connection: sqlite3.Connection, rows: Iterable[dict]) -> None:
    values = [
        (
            int(row["feature_type_id"]),
            row["name"],
            int(row["vector_dim"]),
            row["similarity_metric"],
            row["extraction_params_json"],
            int(row["is_primary"]),
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO feature_types (
            feature_type_id, name, vector_dim, similarity_metric, extraction_params_json, is_primary
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def insert_image_features(
    connection: sqlite3.Connection,
    image_ids: Sequence[int],
    feature_type_map: Dict[str, int],
    feature_arrays: Dict[str, np.ndarray],
) -> None:
    values = []
    for feature_name, matrix in feature_arrays.items():
        feature_type_id = int(feature_type_map[feature_name])
        for row_index, image_id in enumerate(image_ids):
            values.append((int(image_id), feature_type_id, vector_to_json(matrix[row_index])))
    connection.executemany(
        """
        INSERT OR REPLACE INTO image_features (
            image_id, feature_type_id, vector_json
        ) VALUES (?, ?, ?)
        """,
        values,
    )
    connection.commit()


def upsert_experiments(connection: sqlite3.Connection, rows: Iterable[dict]) -> None:
    values = [
        (
            int(row["experiment_id"]),
            row["name"],
            row["feature_set_json"],
            row["weighting_json"],
            row["dataset_version"],
            row["summary_metrics_json"],
            row.get("notes", ""),
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO experiments (
            experiment_id, name, feature_set_json, weighting_json, dataset_version, summary_metrics_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def fetch_feature_type_map(connection: sqlite3.Connection) -> Dict[str, int]:
    rows = connection.execute("SELECT feature_type_id, name FROM feature_types ORDER BY feature_type_id").fetchall()
    return {str(row["name"]): int(row["feature_type_id"]) for row in rows}


def fetch_experiment_rows(connection: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    rows = connection.execute("SELECT * FROM experiments ORDER BY experiment_id").fetchall()
    return {str(row["name"]): row for row in rows}


def ensure_default_experiments(connection: sqlite3.Connection, dataset_version: str) -> None:
    existing = fetch_experiment_rows(connection)
    rows = []
    next_id = max([int(row["experiment_id"]) for row in existing.values()], default=0) + 1
    for name, config in DEFAULT_EXPERIMENTS.items():
        row = existing.get(name)
        experiment_id = int(row["experiment_id"]) if row else next_id
        if row is None:
            next_id += 1
        rows.append(
            {
                "experiment_id": experiment_id,
                "name": name,
                "feature_set_json": json_dumps(sorted(config["weights"].keys())),
                "weighting_json": json_dumps(config["weights"]),
                "dataset_version": dataset_version,
                "summary_metrics_json": row["summary_metrics_json"] if row else json_dumps({}),
                "notes": config.get("notes", ""),
            }
        )
    upsert_experiments(connection, rows)


def get_or_create_query(
    connection: sqlite3.Connection,
    query_source_type: str,
    query_image_path: str,
    preprocess_params_json: str,
) -> int:
    row = connection.execute(
        """
        SELECT query_id
        FROM queries
        WHERE query_source_type = ? AND query_image_path = ?
        """,
        (query_source_type, query_image_path),
    ).fetchone()
    if row:
        connection.execute(
            """
            UPDATE queries
            SET preprocess_params_json = ?
            WHERE query_id = ?
            """,
            (preprocess_params_json, int(row["query_id"])),
        )
        connection.commit()
        return int(row["query_id"])

    cursor = connection.execute(
        """
        INSERT INTO queries (
            query_source_type, query_image_path, created_at_utc, preprocess_params_json
        ) VALUES (?, ?, ?, ?)
        """,
        (query_source_type, query_image_path, utc_now_iso(), preprocess_params_json),
    )
    connection.commit()
    return int(cursor.lastrowid)


def upsert_query_features(
    connection: sqlite3.Connection,
    query_id: int,
    feature_type_map: Dict[str, int],
    query_features: Dict[str, np.ndarray],
) -> None:
    values = [
        (int(query_id), int(feature_type_map[name]), vector_to_json(vector))
        for name, vector in query_features.items()
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO query_features (
            query_id, feature_type_id, vector_json
        ) VALUES (?, ?, ?)
        """,
        values,
    )
    connection.commit()


def create_retrieval_run(
    connection: sqlite3.Connection,
    query_id: int,
    experiment_id: int,
    mode: str,
    top_k: int,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO retrieval_runs (
            query_id, experiment_id, mode, top_k, created_at_utc
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (int(query_id), int(experiment_id), mode, int(top_k), utc_now_iso()),
    )
    connection.commit()
    return int(cursor.lastrowid)


def replace_retrieval_results(connection: sqlite3.Connection, run_id: int, rows: Iterable[dict]) -> None:
    connection.execute("DELETE FROM retrieval_results WHERE run_id = ?", (int(run_id),))
    values = [
        (
            int(run_id),
            int(row["rank"]),
            int(row["image_id"]),
            json_dumps(row["per_feature_scores"]),
            float(row["fused_score"]),
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT INTO retrieval_results (
            run_id, rank, image_id, per_feature_scores_json, fused_score
        ) VALUES (?, ?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def upsert_relevance_judgments(connection: sqlite3.Connection, rows: Iterable[dict]) -> None:
    values = [
        (
            int(row["query_id"]),
            int(row["image_id"]),
            row["judgment_source"],
            int(row["relevance_grade"]),
        )
        for row in rows
    ]
    connection.executemany(
        """
        INSERT OR REPLACE INTO relevance_judgments (
            query_id, image_id, judgment_source, relevance_grade
        ) VALUES (?, ?, ?, ?)
        """,
        values,
    )
    connection.commit()


def update_experiment_summary(
    connection: sqlite3.Connection,
    experiment_id: int,
    summary_metrics: dict,
) -> None:
    connection.execute(
        "UPDATE experiments SET summary_metrics_json = ? WHERE experiment_id = ?",
        (json_dumps(summary_metrics), int(experiment_id)),
    )
    connection.commit()


def load_gallery_rows(connection: sqlite3.Connection, keep_only: bool = True) -> List[sqlite3.Row]:
    if keep_only:
        return connection.execute(
            """
            SELECT *
            FROM images
            WHERE COALESCE(keep, 1) = 1
            ORDER BY image_id
            """
        ).fetchall()
    return connection.execute("SELECT * FROM images ORDER BY image_id").fetchall()


def load_gallery_feature_matrix(
    connection: sqlite3.Connection,
    image_ids: Sequence[int],
    feature_type_id: int,
) -> np.ndarray:
    if not image_ids:
        return np.zeros((0, 0), dtype=np.float32)

    expected_image_ids = [int(image_id) for image_id in image_ids]
    if len(set(expected_image_ids)) != len(expected_image_ids):
        raise ValueError("image_ids must be unique when loading gallery feature matrices.")

    placeholders = ",".join("?" for _ in expected_image_ids)
    rows = connection.execute(
        f"""
        SELECT image_id, vector_json
        FROM image_features
        WHERE feature_type_id = ? AND image_id IN ({placeholders})
        ORDER BY image_id
        """,
        [int(feature_type_id), *expected_image_ids],
    ).fetchall()

    vectors_by_image_id = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in vectors_by_image_id:
            raise ValueError(
                f"Duplicate feature vector for image_id={image_id}, feature_type_id={feature_type_id}."
            )
        vectors_by_image_id[image_id] = parse_vector_json(str(row["vector_json"]))

    missing_image_ids = [image_id for image_id in expected_image_ids if image_id not in vectors_by_image_id]
    if missing_image_ids:
        sample = ", ".join(str(image_id) for image_id in missing_image_ids[:5])
        raise ValueError(
            f"Missing {len(missing_image_ids)} feature vectors for feature_type_id={feature_type_id}; "
            f"first missing image_ids: {sample}"
        )

    vectors = [vectors_by_image_id[image_id] for image_id in expected_image_ids]
    return np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
