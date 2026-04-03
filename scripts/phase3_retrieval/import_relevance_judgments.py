from __future__ import annotations

import argparse
from pathlib import Path

from db_utils import connect_db, upsert_relevance_judgments
from feature_utils import load_csv_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import graded manual relevance judgments into the SQLite database."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument(
        "--judgments-csv",
        default="outputs/manual_relevance/manual_relevance_template.csv",
        help="Reviewed CSV with query_id, candidate_image_id, and relevance_grade filled in.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_csv_rows(Path(args.judgments_csv).resolve())
    import_rows = []
    for row in rows:
        grade_text = str(row.get("relevance_grade", "")).strip()
        if not grade_text:
            continue
        import_rows.append(
            {
                "query_id": int(row["query_id"]),
                "image_id": int(row["candidate_image_id"]),
                "judgment_source": "manual",
                "relevance_grade": int(float(grade_text)),
            }
        )

    if not import_rows:
        raise ValueError("No non-empty relevance_grade values were found in the reviewed CSV.")

    connection = connect_db(Path(args.db_path).resolve())
    try:
        upsert_relevance_judgments(connection, import_rows)
    finally:
        connection.close()

    print(f"Imported manual judgments: {len(import_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
