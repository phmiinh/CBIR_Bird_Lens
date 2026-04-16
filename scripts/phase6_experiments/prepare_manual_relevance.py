from __future__ import annotations

import argparse
from collections import defaultdict
import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from db_utils import connect_db
from feature_utils import DEFAULT_EXPERIMENTS, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a CSV template for manual relevance labeling from stored retrieval results."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=list(DEFAULT_EXPERIMENTS.keys()),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-csv", default="outputs/manual_relevance/manual_relevance_template.csv")
    return parser.parse_args()


def load_latest_runs(connection, experiments: list[str]) -> list[dict]:
    placeholders = ",".join("?" for _ in experiments)
    rows = connection.execute(
        f"""
        SELECT
            runs.run_id,
            runs.query_id,
            runs.experiment_id,
            exp.name AS experiment_name
        FROM retrieval_runs AS runs
        JOIN experiments AS exp ON exp.experiment_id = runs.experiment_id
        WHERE exp.name IN ({placeholders})
        ORDER BY runs.query_id, runs.experiment_id, runs.run_id DESC
        """,
        experiments,
    ).fetchall()

    latest = {}
    for row in rows:
        key = (int(row["query_id"]), int(row["experiment_id"]))
        if key not in latest:
            latest[key] = {
                "run_id": int(row["run_id"]),
                "query_id": int(row["query_id"]),
                "experiment_name": str(row["experiment_name"]),
            }
    return list(latest.values())


def main() -> int:
    args = parse_args()
    connection = connect_db(Path(args.db_path).resolve())
    try:
        latest_runs = load_latest_runs(connection, list(args.experiments))
        rows = []
        for run in latest_runs:
            result_rows = connection.execute(
                """
                SELECT
                    rr.run_id,
                    rr.rank,
                    rr.image_id AS candidate_image_id,
                    q.query_id,
                    q.query_image_path,
                    qi.image_id AS query_image_id,
                    qi.species_name AS query_species_name,
                    qi.processed_relative_path AS query_processed_relative_path,
                    gi.species_name AS candidate_species_name,
                    gi.processed_relative_path AS candidate_processed_relative_path
                FROM retrieval_results AS rr
                JOIN retrieval_runs AS runs ON runs.run_id = rr.run_id
                JOIN queries AS q ON q.query_id = runs.query_id
                LEFT JOIN images AS qi ON qi.processed_relative_path = q.query_image_path
                JOIN images AS gi ON gi.image_id = rr.image_id
                WHERE rr.run_id = ? AND rr.rank <= ?
                ORDER BY rr.rank
                """,
                (int(run["run_id"]), int(args.top_k)),
            ).fetchall()
            for row in result_rows:
                rows.append(
                    {
                        "run_id": int(row["run_id"]),
                        "rank": int(row["rank"]),
                        "candidate_image_id": int(row["candidate_image_id"]),
                        "experiment_name": run["experiment_name"],
                        "query_id": int(row["query_id"]),
                        "query_image_path": str(row["query_image_path"]),
                        "query_image_id": row["query_image_id"],
                        "query_species_name": row["query_species_name"],
                        "query_processed_relative_path": row["query_processed_relative_path"],
                        "candidate_species_name": row["candidate_species_name"],
                        "candidate_processed_relative_path": row["candidate_processed_relative_path"],
                    }
                )
    finally:
        connection.close()

    if not rows:
        raise ValueError("No retrieval results found. Run run_experiments.py first.")

    grouped = defaultdict(lambda: {"experiments": [], "best_rank": None, "row": None})
    for row in rows:
        key = (int(row["query_id"]), int(row["candidate_image_id"]))
        item = grouped[key]
        item["experiments"].append(str(row["experiment_name"]))
        rank = int(row["rank"])
        item["best_rank"] = rank if item["best_rank"] is None else min(item["best_rank"], rank)
        item["row"] = row

    export_rows = []
    for (query_id, candidate_image_id), item in sorted(grouped.items()):
        row = item["row"]
        export_rows.append(
            {
                "query_id": query_id,
                "query_image_id": int(row["query_image_id"]) if row["query_image_id"] is not None else "",
                "query_species_name": str(row["query_species_name"] or ""),
                "query_processed_relative_path": str(row["query_processed_relative_path"] or row["query_image_path"]),
                "candidate_image_id": candidate_image_id,
                "candidate_species_name": str(row["candidate_species_name"] or ""),
                "candidate_processed_relative_path": str(row["candidate_processed_relative_path"]),
                "source_experiments": ",".join(sorted(set(item["experiments"]))),
                "best_rank_seen": int(item["best_rank"]),
                "relevance_grade": "",
                "review_notes": "",
            }
        )

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
            "relevance_grade",
            "review_notes",
        ],
    )
    print(f"Saved manual relevance template: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
