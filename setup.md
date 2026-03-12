# Setup Environment + Kaggle Access

This document covers only environment setup and dataset access.

## 1. Create virtual environment

Run at repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick check:

```powershell
python --version
pip --version
```

## 2. Configure Kaggle API token

1. Sign in to Kaggle.
2. Go to `Settings` -> `API Tokens`.
3. Click `Create Legacy API Key` to download `kaggle.json`.
4. Create token folder if needed:

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
```

5. Copy token file:

```powershell
copy "D:\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json" -Force
```

## 3. Download CUB-200-2011 dataset

Option A - Kaggle API:

```powershell
python scripts/phase1_setup/download_cub.py --force
```

Option B - local zip file:

```powershell
python scripts/phase1_setup/download_cub.py --zip "D:\path\to\cub2002011.zip" --force
```

## 4. Expected output

Dataset root after extraction:

```text
data/raw/cub2002011
```

Check:

```powershell
Get-ChildItem data/raw/cub2002011
```
