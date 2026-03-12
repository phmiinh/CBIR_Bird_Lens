# Project Pipeline

This file describes the execution flow by phase.

## Phase 1 - Prerequisites

Complete [setup.md](setup.md):

- Virtual environment is created and dependencies are installed.
- Dataset is available at `data/raw/cub2002011`.

Start each session with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Phase 2 - Normalization Workflow

### Step 2.1 - Generate review candidates (1000 images)

```powershell
python scripts/phase2_normalize/sample_candidates.py --dataset-root data/raw/cub2002011 --sample-size 1000 --seed 42 --padding-ratio 0.35 --distribution-mode species_balanced
```

Notes:

- `species_balanced` gives a balanced start across species.
- With CUB-200 and `sample-size=1000`, this is typically `5 images/species`.
- Preview images include original with red bbox and normalized crop.

To change padding while preserving reviewed keep/skip states:

```powershell
python scripts/phase2_normalize/sample_candidates.py --dataset-root data/raw/cub2002011 --output-dir outputs/review --padding-ratio 0.35 --reuse-selection-csv outputs/review/candidates.csv --prefill-keep-csv outputs/review/reviewed_candidates.csv
```

Output:

- `outputs/review/candidates.csv`
- `outputs/review/gallery.html`
- `outputs/review/previews/*.jpg`

### Step 2.2 - Manual review in HTML

1. Open `outputs/review/gallery.html`.
2. Use `Species` dropdown to review by class.
3. Each page shows one species.
4. Mark:
- `Keep`: valid image (perching, side view, usable crop).
- `Skip`: invalid image.
- `Reset`: clear status for one image.
5. Click `Download Reviewed CSV`.

### Step 2.3 - Normalize final 500 images

```powershell
python scripts/phase2_normalize/normalize_selected.py --selection-csv outputs/review/reviewed_candidates.csv --dataset-root data/raw/cub2002011 --padding-ratio 0.35 --require-count 500 --max-images 500
```

Output:

- `data/processed/images`
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`
- `outputs/intermediate_examples`

### Step 2.4 - Validation checklist

- Exactly `500` kept images.
- All outputs are `224x224`.
- Metadata includes crop and resize parameters.
- Intermediate examples are available for report.

## Phase 3 - Feature Extraction and Retrieval

### Step 3.1 - Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3.2 - Extract CNN + HSV features

```powershell
python scripts/phase3_retrieval/extract_features.py --metadata-csv data/processed/metadata/images.csv --processed-root data/processed --output-dir data/features --cnn-backbone resnet18 --hsv-bins 8,8,8 --device auto
```

Output:

- `data/features/images_manifest.csv`
- `data/features/cnn_embeddings.npy`
- `data/features/hsv_histograms.npy`
- `data/features/features.jsonl`
- `data/features/config.json`

### Step 3.3 - Build local SQLite feature database

```powershell
python scripts/phase3_retrieval/build_feature_db.py --features-dir data/features --db-path data/features/cbir_features.sqlite --overwrite
```

Tables:

- `images`
- `features`
- `retrieval_logs`

### Step 3.4 - Run top-5 retrieval

Demo UI (upload image and view results):

```powershell
python scripts/phase3_retrieval/demo_ui.py --features-dir data/features --processed-root data/processed --device auto --host 127.0.0.1 --port 7860
```

Then open: `http://127.0.0.1:7860`

Query by `image_id`:

```powershell
python scripts/phase3_retrieval/retrieve_topk.py --features-dir data/features --processed-root data/processed --query-image-id 223 --mode fusion --top-k 5 --w-cnn 0.7 --w-hsv 0.3 --output-csv outputs/retrieval/topk_fusion_id223.csv
```

Query by external image:

```powershell
python scripts/phase3_retrieval/retrieve_topk.py --features-dir data/features --query-image-path "D:\path\to\query.jpg" --mode fusion --top-k 5 --w-cnn 0.7 --w-hsv 0.3 --output-csv outputs/retrieval/topk_fusion_external.csv
```

### Step 3.5 - Evaluate Precision@5

Fusion:

```powershell
python scripts/phase3_retrieval/evaluate_precision_at_k.py --features-dir data/features --mode fusion --k 5 --w-cnn 0.7 --w-hsv 0.3 --output-dir outputs/eval
```

CNN only:

```powershell
python scripts/phase3_retrieval/evaluate_precision_at_k.py --features-dir data/features --mode cnn --k 5 --output-dir outputs/eval
```

HSV only:

```powershell
python scripts/phase3_retrieval/evaluate_precision_at_k.py --features-dir data/features --mode hsv --k 5 --output-dir outputs/eval
```
