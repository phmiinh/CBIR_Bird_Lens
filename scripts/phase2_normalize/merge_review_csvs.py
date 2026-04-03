from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple reviewed candidate CSV files into one combined selection CSV."
    )
    parser.add_argument(
        "--input-csv",
        action="append",
        required=True,
        help="Reviewed or candidate CSV to merge. Repeat this flag for each batch in order.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/review/final_reviewed_candidates.csv",
        help="Merged CSV path used by normalize_selected.py.",
    )
    return parser.parse_args()


def read_csv_rows(file_path: Path) -> tuple[list[str], list[dict]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"No CSV header found in {file_path}")
    return fieldnames, rows


def parse_image_id(row: dict, file_path: Path) -> int:
    raw = str(row.get("image_id", "")).strip()
    if not raw:
        raise ValueError(f"Missing image_id in {file_path}")
    return int(float(raw))


def main() -> int:
    args = parse_args()
    input_paths = [Path(value).resolve() for value in args.input_csv]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Could not find input CSV: {path}")

    merged_rows: list[dict] = []
    merged_fieldnames: list[str] | None = None
    seen_image_ids: set[int] = set()

    for file_path in input_paths:
        fieldnames, rows = read_csv_rows(file_path)
        if merged_fieldnames is None:
            merged_fieldnames = fieldnames
        elif fieldnames != merged_fieldnames:
            raise ValueError(
                f"CSV schema mismatch. Expected fields from first file, but {file_path} has a different header."
            )

        for row in rows:
            image_id = parse_image_id(row, file_path)
            if image_id in seen_image_ids:
                raise ValueError(
                    f"Duplicate image_id {image_id} detected while merging. "
                    f"Check overlapping batches before rerunning."
                )
            seen_image_ids.add(image_id)
            merged_rows.append(row)

    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames or [])
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Merged {len(input_paths)} CSV file(s) into: {output_csv}")
    print(f"Total unique image rows: {len(merged_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
