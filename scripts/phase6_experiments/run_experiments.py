from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from db_utils import upsert_relevance_judgments
from feature_utils import write_csv_rows
from retrieval_core import RetrievalEngine


EXPERIMENT_NAMES = [
    "handcrafted_only",
    "cnn_only",
    "fusion",
    "ablation_no_regional_color",
    "ablation_no_shape",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the registered retrieval experiments on a diverse gallery query subset."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--query-count", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="outputs/experiments")
    return parser.parse_args()


def sample_diverse_query_ids(gallery_rows, query_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_species = defaultdict(list)
    for row in gallery_rows:
        by_species[int(row["species_id"])].append(int(row["image_id"]))

    species_ids = list(by_species.keys())
    rng.shuffle(species_ids)
    for image_ids in by_species.values():
        rng.shuffle(image_ids)

    selected = []
    species_cursor = 0
    while len(selected) < query_count and species_ids:
        species_id = species_ids[species_cursor % len(species_ids)]
        bucket = by_species[species_id]
        if bucket:
            selected.append(bucket.pop())
        species_cursor += 1
        if species_cursor > len(species_ids) * max(1, query_count) and not any(by_species.values()):
            break
    return selected[:query_count]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = RetrievalEngine(
        db_path=Path(args.db_path),
        processed_root=Path(args.processed_root),
        device=args.device,
    )
    try:
        query_image_ids = sample_diverse_query_ids(engine.gallery_rows, args.query_count, args.seed)
        if len(query_image_ids) < args.query_count:
            print(f"Warning: only sampled {len(query_image_ids)} query images.")

        query_rows = []
        run_rows = []
        query_id_by_image_id = {}

        for query_order, query_image_id in enumerate(query_image_ids, start=1):
            gallery_row = engine.gallery_by_image_id[query_image_id]
            for experiment_name in EXPERIMENT_NAMES:
                payload = engine.run_retrieval(
                    experiment_name=experiment_name,
                    top_k=args.top_k,
                    query_image_id=query_image_id,
                    persist=True,
                )
                query_id_by_image_id[query_image_id] = int(payload["query_id"])
                run_rows.append(
                    {
                        "query_id": int(payload["query_id"]),
                        "query_image_id": query_image_id,
                        "experiment_name": experiment_name,
                        "run_id": int(payload["run_id"]),
                        "top_k": int(args.top_k),
                    }
                )

            query_rows.append(
                {
                    "query_order": query_order,
                    "query_id": query_id_by_image_id[query_image_id],
                    "query_image_id": query_image_id,
                    "species_id": int(gallery_row["species_id"]),
                    "species_name": str(gallery_row["species_name"]),
                    "processed_relative_path": str(gallery_row["processed_relative_path"]),
                }
            )

        species_proxy_rows = []
        for query_row in query_rows:
            query_species_id = int(query_row["species_id"])
            for gallery_row in engine.gallery_rows:
                species_proxy_rows.append(
                    {
                        "query_id": int(query_row["query_id"]),
                        "image_id": int(gallery_row["image_id"]),
                        "judgment_source": "species_proxy",
                        "relevance_grade": 1 if int(gallery_row["species_id"]) == query_species_id else 0,
                    }
                )
        upsert_relevance_judgments(engine.connection, species_proxy_rows)
    finally:
        engine.close()

    write_csv_rows(
        output_dir / "query_subset.csv",
        query_rows,
        ["query_order", "query_id", "query_image_id", "species_id", "species_name", "processed_relative_path"],
    )
    write_csv_rows(
        output_dir / "retrieval_runs.csv",
        run_rows,
        ["query_id", "query_image_id", "experiment_name", "run_id", "top_k"],
    )

    print(f"Saved query subset:   {output_dir / 'query_subset.csv'}")
    print(f"Saved experiment runs:{output_dir / 'retrieval_runs.csv'}")
    print(f"Species-proxy labels inserted for {len(query_rows)} queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
