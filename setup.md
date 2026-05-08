# Environment Setup

This file covers only:
- Python environment setup
- dependency installation
- Kaggle API access
- CUB-200-2011 download

For the working execution flow, continue with [roadmap.md](/d:/PTIT/Multimedia%20Database/roadmap.md).

Current local workspace note:
- the project virtual environment `.venv` is present
- raw CUB data is present under `data/raw/cub2002011`
- downstream pipeline status is tracked in [current_status_review.md](/d:/PTIT/Multimedia%20Database/current_status_review.md)

## 1. Create and Activate the Virtual Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip` and install project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment includes:
- normalization dependencies
- `torch` / `torchvision` for the secondary CNN descriptor
- `scikit-image` for LBP and HOG
- `gradio` for the optional demo UI

## 2. Configure Kaggle Access

Generate `kaggle.json` from your Kaggle account settings, then move it to:

```text
C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
```

If you already downloaded `kaggle.json` into `Downloads`, copy it with:

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.kaggle"
Copy-Item "D:\Downloads\kaggle.json" "$HOME\.kaggle\kaggle.json" -Force
```

Optional verification:

```powershell
python -c "import kaggle; print('Kaggle API ready')"
```

## 3. Download the CUB Dataset

Run:

```powershell
python scripts/phase1_setup/download_cub.py
```

Expected output root:

```text
data/raw/cub2002011
```

The script supports the Kaggle-hosted CUB package used in this repository and resolves the extracted dataset root automatically.

## 4. Confirm the Raw Dataset Layout

You should have a structure similar to:

```text
data/raw/cub2002011/
  CUB_200_2011/
    images.txt
    image_class_labels.txt
    train_test_split.txt
    bounding_boxes.txt
    images/
```

Once this is ready, move to [roadmap.md](/d:/PTIT/Multimedia%20Database/roadmap.md).
