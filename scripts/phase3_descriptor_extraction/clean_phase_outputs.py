from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PHASE3_TARGETS = [
    Path("data/features"),
    Path("outputs/retrieval"),
    Path("outputs/eval"),
    Path("outputs/experiments"),
    Path("outputs/manual_relevance"),
    Path("outputs/report_artifacts"),
    Path("outputs/demo_ui_queries"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove old Phase 3+ downstream artifacts before rebuilding features, DB, retrieval logs, and evaluation outputs."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the Phase 3+ downstream artifact paths. Without this flag, the script only prints the cleanup plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    targets = [(root / relative_path).resolve() for relative_path in PHASE3_TARGETS]

    print("Phase 3+ downstream cleanup targets:")
    for target in targets:
        exists = "exists" if target.exists() else "missing"
        print(f"- {target} [{exists}]")

    print("This cleanup intentionally keeps:")
    print(f"- {(root / 'data/raw').resolve()}")
    print(f"- {(root / 'data/processed').resolve()}")
    print(f"- {(root / 'outputs/review').resolve()}")
    print(f"- {(root / 'outputs/review_batch_2').resolve()}")

    if not args.yes:
        print("Dry run only. Re-run with --yes to delete the old Phase 3+ downstream artifacts.")
        return 0

    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    (root / "data/features").mkdir(parents=True, exist_ok=True)
    print("Old Phase 3+ downstream artifacts deleted. data/features has been recreated as an empty directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
