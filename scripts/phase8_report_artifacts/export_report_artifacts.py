from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from db_utils import connect_db
from feature_utils import build_descriptor_table_rows, load_json, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export report-ready artifacts from the DB-first CBIR pipeline."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--output-dir", default="outputs/report_artifacts")
    parser.add_argument("--example-experiment", default="fusion")
    return parser.parse_args()


def write_block_diagram(file_path: Path) -> None:
    content = """# System Block Diagram

```mermaid
flowchart LR
    A[Raw CUB Images + Bounding Boxes] --> B[Phase 2 Preprocessing]
    B --> C[Normalized 224x224 Gallery]
    C --> D[Feature Extraction]
    D --> E[SQLite Descriptor System of Record]
    Q[Query Image] --> R[Query Preprocessing]
    R --> S[Query Feature Extraction]
    S --> E
    E --> T[Application-Layer Exhaustive kNN]
    T --> U[Score Fusion + Ranking]
    U --> V[Top-5 Retrieval Results]
    V --> W[Evaluation and Experiment Tracking]
    W --> E
```

SQLite stores metadata, descriptor definitions, vectors, query cache, run logs, judgments, and experiment summaries.
The similarity scan and score fusion run in application code; SQLite is not treated as a native vector-search engine.
"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    features_dir = Path(args.features_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(features_dir / "config.json")
    descriptor_rows = build_descriptor_table_rows({str(k): int(v) for k, v in config["feature_dims"].items()})
    write_csv_rows(
        output_dir / "descriptor_table.csv",
        descriptor_rows,
        [
            "feature_name",
            "display_name",
            "vector_dim",
            "similarity_metric",
            "is_primary",
            "information_value",
            "dimension_derivation",
            "extraction_params_json",
            "strengths",
            "weaknesses",
        ],
    )

    write_block_diagram(output_dir / "system_block_diagram.md")

    connection = connect_db(Path(args.db_path).resolve())
    try:
        experiment_rows = connection.execute("SELECT * FROM experiments ORDER BY experiment_id").fetchall()
        summary_rows = []
        for row in experiment_rows:
            summary_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "name": str(row["name"]),
                    "feature_set_json": str(row["feature_set_json"]),
                    "weighting_json": str(row["weighting_json"]),
                    "score_normalization": str(
                        row["score_normalization"] if "score_normalization" in row.keys() else "raw"
                    ),
                    "summary_metrics_json": str(row["summary_metrics_json"]),
                    "notes": str(row["notes"] or ""),
                }
            )
        write_csv_rows(
            output_dir / "experiment_summaries.csv",
            summary_rows,
            [
                "experiment_id",
                "name",
                "feature_set_json",
                "weighting_json",
                "score_normalization",
                "summary_metrics_json",
                "notes",
            ],
        )

        example_row = connection.execute(
            """
            SELECT
                q.query_id,
                q.query_image_path,
                rr.rank,
                rr.image_id,
                rr.per_feature_scores_json,
                rr.fused_score
            FROM retrieval_runs AS runs
            JOIN experiments AS exp ON exp.experiment_id = runs.experiment_id
            JOIN queries AS q ON q.query_id = runs.query_id
            JOIN retrieval_results AS rr ON rr.run_id = runs.run_id
            WHERE exp.name = ?
            ORDER BY runs.run_id DESC, rr.rank ASC
            LIMIT 1
            """,
            (args.example_experiment,),
        ).fetchone()
    finally:
        connection.close()

    if example_row:
        example_payload = {
            "query_id": int(example_row["query_id"]),
            "query_image_path": str(example_row["query_image_path"]),
            "rank": int(example_row["rank"]),
            "image_id": int(example_row["image_id"]),
            "per_feature_scores": json.loads(str(example_row["per_feature_scores_json"])),
            "fused_score": float(example_row["fused_score"]),
        }
        (output_dir / "example_retrieval_breakdown.json").write_text(
            json.dumps(example_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    print(f"Saved report artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
