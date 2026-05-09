from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the DB-backed downstream CBIR pipeline from feature artifacts through "
            "experiments, relevance proxies, evaluation, and report artifacts."
        )
    )
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--query-count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-db-rebuild", action="store_true")
    parser.add_argument("--skip-auto-relevance", action="store_true")
    parser.add_argument("--import-manual-if-filled", action="store_true")
    return parser.parse_args()


def run_step(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def has_filled_relevance_grade(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("relevance_grade", "")).strip():
                return True
    return False


def print_final_summary(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        print("\n=== Final DB Summary ===")
        for table in [
            "images",
            "feature_types",
            "image_features",
            "queries",
            "query_features",
            "experiments",
            "retrieval_runs",
            "retrieval_results",
            "relevance_judgments",
        ]:
            count = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
            print(f"{table}: {count}")

        sources = connection.execute(
            """
            SELECT judgment_source, COUNT(*) AS count
            FROM relevance_judgments
            GROUP BY judgment_source
            ORDER BY judgment_source
            """
        ).fetchall()
        print("judgment_sources:")
        for row in sources:
            print(f"  {row['judgment_source']}: {row['count']}")

        summaries = connection.execute(
            """
            SELECT name, summary_metrics_json
            FROM experiments
            ORDER BY experiment_id
            """
        ).fetchall()
        print("experiment_metrics:")
        for row in summaries:
            metrics = json.loads(str(row["summary_metrics_json"] or "{}"))
            print(f"  {row['name']}: {json.dumps(metrics, ensure_ascii=True)}")
    finally:
        connection.close()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    python = sys.executable

    features_dir = Path(args.features_dir)
    db_path = Path(args.db_path)
    processed_root = Path(args.processed_root)
    manual_template = Path("outputs/manual_relevance/manual_relevance_template.csv")
    auto_relevance_csv = Path("outputs/auto_relevance/auto_visual_relevance.csv")
    manual_import_csv = root / manual_template

    if args.import_manual_if_filled and has_filled_relevance_grade(root / manual_template):
        manual_backup = root / "outputs/manual_relevance/manual_relevance_filled_backup.csv"
        manual_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / manual_template, manual_backup)
        manual_import_csv = manual_backup
        print(f"Preserved filled manual relevance CSV before rerun: {manual_backup}", flush=True)

    if not args.skip_db_rebuild:
        run_step(
            "Phase 4 - rebuild SQLite feature database",
            [
                python,
                "scripts/phase4_feature_database/build_feature_db.py",
                "--features-dir",
                str(features_dir),
                "--db-path",
                str(db_path),
                "--overwrite",
            ],
            root,
        )

    run_step(
        "Phase 6 - run retrieval experiments and species proxy judgments",
        [
            python,
            "scripts/phase6_experiments/run_experiments.py",
            "--db-path",
            str(db_path),
            "--processed-root",
            str(processed_root),
            "--query-count",
            str(args.query_count),
            "--top-k",
            str(args.top_k),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--output-dir",
            "outputs/experiments",
        ],
        root,
    )

    run_step(
        "Phase 6.3 - export manual relevance template",
        [
            python,
            "scripts/phase6_experiments/prepare_manual_relevance.py",
            "--db-path",
            str(db_path),
            "--top-k",
            str(args.top_k),
            "--output-csv",
            str(manual_template),
        ],
        root,
    )

    if not args.skip_auto_relevance:
        run_step(
            "Phase 6.4 - generate auto visual proxy relevance",
            [
                python,
                "scripts/phase6_experiments/generate_auto_relevance_judgments.py",
                "--db-path",
                str(db_path),
                "--template-csv",
                str(manual_template),
                "--output-csv",
                str(auto_relevance_csv),
            ],
            root,
        )
        run_step(
            "Phase 6.5 - import auto visual proxy relevance",
            [
                python,
                "scripts/phase6_experiments/import_relevance_judgments.py",
                "--db-path",
                str(db_path),
                "--judgments-csv",
                str(auto_relevance_csv),
                "--judgment-source",
                "auto_visual_proxy",
            ],
            root,
        )

    if args.import_manual_if_filled and has_filled_relevance_grade(manual_import_csv):
        run_step(
            "Phase 6.6 - import filled manual relevance",
            [
                python,
                "scripts/phase6_experiments/import_relevance_judgments.py",
                "--db-path",
                str(db_path),
                "--judgments-csv",
                str(manual_import_csv.relative_to(root)),
                "--judgment-source",
                "manual",
            ],
            root,
        )

    run_step(
        "Phase 7 - evaluate all available judgment sources",
        [
            python,
            "scripts/phase7_evaluation/evaluate_retrieval.py",
            "--db-path",
            str(db_path),
            "--judgment-source",
            "all",
            "--k",
            str(args.top_k),
            "--output-dir",
            "outputs/eval",
        ],
        root,
    )

    run_step(
        "Phase 8 - export report artifacts",
        [
            python,
            "scripts/phase8_report_artifacts/export_report_artifacts.py",
            "--db-path",
            str(db_path),
            "--features-dir",
            str(features_dir),
            "--output-dir",
            "outputs/report_artifacts",
            "--example-experiment",
            "fusion",
        ],
        root,
    )

    print_final_summary(root / db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
