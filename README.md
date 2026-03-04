# Bird Image Storage and Retrieval - Data Collection and Normalization

This workspace covers the dataset ingestion and image normalization stage for a CBIR assignment built on top of CUB-200-2011.

## 1. Environment setup

Create a virtual environment and install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Kaggle access

Option A: Kaggle API

1. Log in to Kaggle.
2. Go to `Account` -> `Create New Token`.
3. Place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`.

Then run:

```powershell
python scripts/download_cub.py --force
```

Option B: manual zip download

1. Download the dataset zip from https://www.kaggle.com/datasets/wenewone/cub2002011
2. Run:

```powershell
python scripts/download_cub.py --zip "D:\path\to\cub2002011.zip" --force
```

Expected output root:

```text
data/raw/cub2002011
```

## 3. Create a manual review batch

Generate 1000 assignment-oriented candidates plus an HTML gallery for manual filtering:

```powershell
python scripts/sample_candidates.py --dataset-root data/raw/cub2002011 --sample-size 1000 --seed 42
```

The default `assignment` sampling strategy is not plain random. It uses CUB attributes and part visibility to enrich the review batch toward:

- `has_shape::perching-like`
- likely side-view images where only one eye / one wing is visible
- images with visible legs and a wider-than-tall bird box

If you want the old behavior, you can still force pure random sampling:

```powershell
python scripts/sample_candidates.py --dataset-root data/raw/cub2002011 --sample-size 1000 --seed 42 --sampling-strategy random
```

This creates:

- `outputs/review/candidates.csv`
- `outputs/review/gallery.html`
- `outputs/review/previews/*.jpg`

Open `gallery.html` in a browser. The review board shows 20 images per page and lets you mark each one as `Keep` or `Skip`, then export a reviewed CSV directly from the browser.

The exported CSV will contain:

- `keep=1` for selected images
- `keep=0` for skipped images
- empty `keep` for unreviewed images

Review only the images that match:

- bird is perching
- bird is shown from roughly side view
- the crop still looks usable after normalization

If you still prefer manual spreadsheet editing, you can edit `candidates.csv` directly and set `keep=1`.

## 4. Normalize the final selection

After you finish review and have exactly 500 kept rows:

```powershell
python scripts/normalize_selected.py --selection-csv outputs/review/candidates.csv --dataset-root data/raw/cub2002011 --require-count 500 --max-images 500
```

If you exported a reviewed CSV from the HTML page into another folder, pass that path to `--selection-csv` instead.

This exports:

- normalized 224x224 images in `data/processed/images`
- metadata in `data/processed/metadata/images.csv`
- metadata in `data/processed/metadata/images.jsonl`
- report-ready intermediate examples in `outputs/intermediate_examples`

## 5. Normalization rule used

The implemented preprocessing rule is:

1. Read the original image.
2. Use the CUB bounding box as the bird ROI.
3. Expand that ROI into a square crop with configurable padding (`padding_ratio`).
4. Resize the crop to `224x224`.
5. Store both image-level metadata and preprocessing parameters.

This square-crop approach avoids the stronger shape distortion that would happen if a non-square bounding box were resized directly into `224x224`.
