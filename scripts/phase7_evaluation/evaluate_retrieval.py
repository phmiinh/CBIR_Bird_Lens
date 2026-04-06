from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from db_utils import connect_db, update_experiment_summary
from feature_utils import save_json, write_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DB-backed retrieval runs with manual relevance and/or species proxy."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument(
        "--judgment-source",
        choices=("manual", "species_proxy", "both"),
        default="both",
        help="Which relevance source to evaluate.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser.parse_args()


def dcg_at_k(grades: list[int], k: int) -> float:
    total = 0.0
    for rank, grade in enumerate(grades[:k], start=1):
        gain = (2**int(grade)) - 1
        total += gain / math.log2(rank + 1)
    return total


def precision_at_k(grades: list[int], k: int, relevant_threshold: int = 1) -> float:
    if k <= 0:
        return 0.0
    relevant = sum(1 for grade in grades[:k] if int(grade) >= relevant_threshold)
    return float(relevant / k)


def load_latest_runs(connection):
    rows = connection.execute(
        """
        SELECT
            runs.run_id,
            runs.query_id,
            runs.experiment_id,
            runs.top_k,
            exp.name AS experiment_name
        FROM retrieval_runs AS runs
        JOIN experiments AS exp ON exp.experiment_id = runs.experiment_id
        ORDER BY runs.query_id, runs.experiment_id, runs.run_id DESC
        """
    ).fetchall()

    latest = {}
    for row in rows:
        key = (int(row["query_id"]), int(row["experiment_id"]))
        if key not in latest:
            latest[key] = row
    return list(latest.values())


def load_judgments(connection, judgment_source: str):
    rows = connection.execute(
        """
        SELECT query_id, image_id, relevance_grade
        FROM relevance_judgments
        WHERE judgment_source = ?
        """,
        (judgment_source,),
    ).fetchall()
    by_query = defaultdict(dict)
    for row in rows:
        by_query[int(row["query_id"])][int(row["image_id"])] = int(row["relevance_grade"])
    return by_query


def evaluate_source(connection, judgment_source: str, k: int, output_dir: Path) -> dict:
    latest_runs = load_latest_runs(connection)
    judgments_by_query = load_judgments(connection, judgment_source)
    if not judgments_by_query:
        return {}

    per_query_rows = []
    by_experiment = defaultdict(list)

    for run in latest_runs:
        query_id = int(run["query_id"])
        if judgment_source == "manual" and query_id not in judgments_by_query:
            continue

        results = connection.execute(
            """
            SELECT image_id, rank, fused_score
            FROM retrieval_results
            WHERE run_id = ?
            ORDER BY rank
            """,
            (int(run["run_id"]),),
        ).fetchall()
        grades = [int(judgments_by_query.get(query_id, {}).get(int(row["image_id"]), 0)) for row in results[:k]]
        precision = precision_at_k(grades, k)

        metric_row = {
            "experiment_name": str(run["experiment_name"]),
            "query_id": query_id,
            "run_id": int(run["run_id"]),
            "precision_at_k": precision,
            "judged_in_topk": sum(1 for row in results[:k] if int(row["image_id"]) in judgments_by_query.get(query_id, {})),
        }

        if judgment_source == "manual":
            ideal_grades = sorted(judgments_by_query[query_id].values(), reverse=True)
            ideal_dcg = dcg_at_k(ideal_grades, k)
            ndcg = (dcg_at_k(grades, k) / ideal_dcg) if ideal_dcg > 0 else 0.0
            metric_row["ndcg_at_k"] = ndcg
            by_experiment[str(run["experiment_name"])].append((precision, ndcg))
        else:
            by_experiment[str(run["experiment_name"])].append((precision,))

        per_query_rows.append(metric_row)

    fieldnames = ["experiment_name", "query_id", "run_id", "precision_at_k", "judged_in_topk"]
    if judgment_source == "manual":
        fieldnames.append("ndcg_at_k")
    write_csv_rows(output_dir / f"{judgment_source}_per_query.csv", per_query_rows, fieldnames)

    summary = {}
    for experiment_name, values in by_experiment.items():
        if judgment_source == "manual":
            precisions = [item[0] for item in values]
            ndcgs = [item[1] for item in values]
            summary[experiment_name] = {
                "evaluated_queries": len(values),
                "manual_precision_at_k": float(sum(precisions) / len(precisions)) if values else 0.0,
                "manual_ndcg_at_k": float(sum(ndcgs) / len(ndcgs)) if values else 0.0,
                "k": int(k),
            }
        else:
            precisions = [item[0] for item in values]
            summary[experiment_name] = {
                "evaluated_queries": len(values),
                "species_proxy_precision_at_k": float(sum(precisions) / len(precisions)) if values else 0.0,
                "k": int(k),
            }

    save_json(output_dir / f"{judgment_source}_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_sources = ["manual", "species_proxy"] if args.judgment_source == "both" else [args.judgment_source]

    connection = connect_db(Path(args.db_path).resolve())
    try:
        experiment_rows = connection.execute("SELECT experiment_id, name, summary_metrics_json FROM experiments").fetchall()
        experiment_map = {
            str(row["name"]): {
                "experiment_id": int(row["experiment_id"]),
                "summary_metrics": json.loads(str(row["summary_metrics_json"] or "{}")),
            }
            for row in experiment_rows
        }

        source_summaries = {}
        for source in requested_sources:
            summary = evaluate_source(connection, source, args.k, output_dir)
            source_summaries[source] = summary

        for experiment_name, info in experiment_map.items():
            merged = dict(info["summary_metrics"])
            for source in requested_sources:
                if experiment_name in source_summaries.get(source, {}):
                    merged[source] = source_summaries[source][experiment_name]
            update_experiment_summary(connection, info["experiment_id"], merged)
    finally:
        connection.close()

    print(f"Saved evaluation reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
