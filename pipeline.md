# Project Pipeline

Tai lieu nay mo ta luong thuc hien bai tap theo tung phase.

## Phase 1 - Prerequisites

Hoan thanh phan chuan bi trong [setup.md](setup.md):

- Da tao `.venv` va cai dependencies.
- Da co dataset tai `data/raw/cub2002011`.

Moi lan bat dau phien lam viec:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Phase 2 - Normalization Workflow

### Step 2.1 - Generate review candidates (1000 images)

```powershell
python scripts/sample_candidates.py --dataset-root data/raw/cub2002011 --sample-size 1000 --seed 42 --padding-ratio 0.35 --distribution-mode species_balanced
```

Y nghia chinh:

- `species_balanced`: phan bo deu theo loai.
- Voi CUB-200 va `sample-size=1000`: thuong la `5 anh / loai`.
- Preview gom anh goc co bbox do va anh crop vuong de danh gia nhanh.

Neu ban chi muon doi `padding-ratio` ma giu nguyen danh sach anh va trang thai `Keep/Skip` da review:

```powershell
python scripts/sample_candidates.py --dataset-root data/raw/cub2002011 --output-dir outputs/review --padding-ratio 0.35 --reuse-selection-csv outputs/review/candidates.csv --prefill-keep-csv outputs/review/reviewed_candidates.csv
```

Output:

- `outputs/review/candidates.csv`
- `outputs/review/gallery.html`
- `outputs/review/previews/*.jpg`

### Step 2.2 - Manual review in HTML

1. Mo `outputs/review/gallery.html`.
2. Dung dropdown `Species` de review theo tung loai.
3. Moi trang tuong ung dung 1 loai (species page).
4. Gan nhan:
- `Keep`: anh dat dieu kien (perching, side view, crop usable).
- `Skip`: anh khong dat.
- `Reset`: xoa trang thai cua anh.
5. Bam `Download Reviewed CSV` de xuat file CSV da review.

Luu y:

- Trang thai review duoc luu bang local storage cua browser.
- `Clear Local Review` se xoa toan bo trang thai local.

### Step 2.3 - Normalize final 500 images

Sau khi co du 500 anh `Keep`, chay:

```powershell
python scripts/normalize_selected.py --selection-csv outputs/review/reviewed_candidates.csv --dataset-root data/raw/cub2002011 --padding-ratio 0.35 --require-count 500 --max-images 500
```

Output:

- `data/processed/images` (anh chuan hoa 224x224).
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`
- `outputs/intermediate_examples` (ket qua trung gian cho bao cao).

### Step 2.4 - Validation checklist

- Dung `500` anh duoc chon (`--require-count 500`).
- Tat ca anh dau ra co kich thuoc `224x224`.
- Metadata co thong tin crop + resize.
- Co du vi du before/after trong `outputs/intermediate_examples`.

## Phase 3 - Feature Extraction and Retrieval Prep

Day la buoc tiep theo sau khi normalize xong.

### Step 3.1 - Define feature set

Can trich xuat it nhat 2 nhom:

- Similarity feature: `CNN embedding` (pretrained model).
- Difference feature: `HSV histogram` (mau) hoac texture/shape descriptor.

### Step 3.2 - Build feature storage schema

Tao schema de luu ket qua trich xuat:

- `images`: image_id, path, species, preprocessing params.
- `features`: image_id, feature_type, vector, dim.
- `retrieval_logs` (optional): query_id, topk_ids, scores, config.

### Step 3.3 - Implement extraction scripts (next coding task)

Muc tieu script:

- Doc anh tu `data/processed/images`.
- Tinh `cnn_embedding` va `hsv_hist`.
- Luu output ra file de ingest DB (CSV/JSONL/NPY).

### Step 3.4 - Retrieval pipeline

Muc tieu:

- Query image -> preprocess giong dataset.
- Extract cung loai feature.
- Tinh similarity (cosine/euclidean), fusion score neu can.
- Sap xep va tra ve `top-5`.

### Step 3.5 - Evaluation

- Tinh `Precision@5` cho bo query.
- So sanh cau hinh: HSV-only, CNN-only, fusion.
- Ghi lai failure cases de dua vao bao cao.
