from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

import gradio as gr

SHARED_DIR = Path(__file__).resolve().parents[1] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))

from feature_utils import resolve_processed_image_path
from retrieval_core import RetrievalEngine

DEMO_EXPERIMENTS = ("calibrated_handcrafted", "fusion", "cnn_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the optional Gradio demo UI on top of the DB-backed retrieval core."
    )
    parser.add_argument("--db-path", default="data/features/cbir_features.sqlite")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def build_app(engine: RetrievalEngine) -> gr.Blocks:
    temp_dir = (Path("outputs") / "demo_ui_queries").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    registered_experiments = list(engine.experiment_rows.keys())
    experiment_names = [name for name in DEMO_EXPERIMENTS if name in engine.experiment_rows]
    if not experiment_names:
        experiment_names = registered_experiments
    if not experiment_names:
        raise ValueError("No experiments are registered in the SQLite database.")
    default_experiment = experiment_names[0]

    def run_demo(query_image, experiment_name, top_k):
        if query_image is None:
            return [], {"error": "Please upload a query image."}

        temp_path = temp_dir / f"query_{uuid4().hex}.png"
        query_image.save(temp_path)
        payload = engine.run_retrieval(
            experiment_name=experiment_name,
            top_k=int(top_k),
            query_image_path=str(temp_path),
            # Gradio callbacks run in worker threads. For the demo UI we only
            # need live retrieval results, not DB logging, so avoid cross-thread
            # SQLite writes here.
            persist=False,
        )

        gallery_items = []
        for row in payload["results"]:
            image_path = resolve_processed_image_path(row, engine.processed_root)
            caption = f"#{row['rank']} | id={row['image_id']} | {row['species_name']} | score={row['fused_score']:.4f}"
            gallery_items.append((str(image_path), caption))
        return gallery_items, payload

    with gr.Blocks(title="Bird CBIR Demo", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Bird Image Retrieval Demo")
        gr.Markdown(
            "Upload a bird image, choose an experiment configuration, and inspect the DB-backed top-k retrieval results."
        )
        with gr.Row():
            query_input = gr.Image(label="Query Image", type="pil")
            with gr.Column():
                experiment_input = gr.Dropdown(
                    choices=experiment_names,
                    value=default_experiment,
                    label="Experiment",
                )
                topk_input = gr.Slider(1, 10, value=5, step=1, label="Top K")
                run_btn = gr.Button("Run Retrieval", variant="primary")

        gallery_output = gr.Gallery(label="Top-K Results", columns=5, rows=1, height=360, object_fit="contain")
        payload_output = gr.JSON(label="Retrieval Payload")

        run_btn.click(run_demo, inputs=[query_input, experiment_input, topk_input], outputs=[gallery_output, payload_output])

    return app


def main() -> int:
    args = parse_args()
    engine = RetrievalEngine(
        db_path=Path(args.db_path),
        processed_root=Path(args.processed_root),
        device=args.device,
    )
    # Gradio executes callbacks in worker threads. Preload gallery matrices in
    # the main thread so live demo callbacks do not touch the SQLite connection.
    engine.preload_feature_matrices()
    app = build_app(engine)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
