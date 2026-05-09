# Demo Run Guideline

This file only covers how to run the retrieval demo.

## Prerequisites

Run commands from the repository root:

```powershell
cd "d:\PTIT\Multimedia Database"
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

The demo expects these artifacts to already exist:

- `data/features/cbir_features.sqlite`
- `data/processed/images/`
- `data/processed/metadata/images.csv`

## Launch The Web Demo

```powershell
python scripts/phase5_retrieval/demo_ui.py `
  --db-path data/features/cbir_features.sqlite `
  --processed-root data/processed `
  --device cpu `
  --host 127.0.0.1 `
  --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

In the UI:

1. Upload a bird image.
2. Choose an experiment configuration.
3. Keep `Top K = 5` for the report demo.
4. Click `Run Retrieval`.
5. Inspect the image gallery and JSON payload.

Recommended demo experiments:

- `calibrated_handcrafted` - main handcrafted CBIR mode for the report demo
- `fusion` - handcrafted-first mode with secondary CNN support
- `cnn_only` - semantic CNN baseline for comparison

The Gradio UI uses live retrieval with `persist=False`, so it does not write retrieval runs during UI interaction.
