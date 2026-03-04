# Project Pipeline

Tài liệu này mô tả luồng chạy theo từng phase của bài tập.

## Phase 1 - Prerequisites

Hoàn thành phần chuẩn bị trong [setup.md](setup.md):

- Đã tạo `.venv` và cài dependencies.
- Đã có dataset tại `data/raw/cub2002011`.

Mỗi lần bắt đầu phiên làm việc:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Phase 2 - Normalization Workflow

### Step 2.1 - Generate review candidates (1000 images)

```powershell
python scripts/sample_candidates.py --dataset-root data/raw/cub2002011 --sample-size 1000 --seed 42 --padding-ratio 0.2 --distribution-mode species_balanced
```

Ý nghĩa chính:

- `species_balanced`: phân bố đều theo loài.
- Với CUB-200 và `sample-size=1000`: thường là `5 ảnh / loài`.
- Preview gồm ảnh gốc có bbox đỏ và ảnh crop vuông để đánh giá nhanh.

Output:

- `outputs/review/candidates.csv`
- `outputs/review/gallery.html`
- `outputs/review/previews/*.jpg`

### Step 2.2 - Manual review in HTML

1. Mở `outputs/review/gallery.html`.
2. Dùng dropdown `Species` để review theo từng loài.
3. Mỗi trang tương ứng đúng 1 loài (species page).
4. Gắn nhãn:
   - `Keep`: ảnh đạt điều kiện (perching, side view, crop usable).
   - `Skip`: ảnh không đạt.
   - `Reset`: xóa trạng thái của ảnh.
5. Bấm `Download Reviewed CSV` để xuất file CSV đã review.

Lưu ý:

- Trạng thái review được lưu bằng local storage của browser.
- `Clear Local Review` sẽ xóa toàn bộ trạng thái local.

### Step 2.3 - Normalize final 500 images

Sau khi có đủ 500 ảnh `Keep`, chạy:

```powershell
python scripts/normalize_selected.py --selection-csv outputs/review/candidates.csv --dataset-root data/raw/cub2002011 --padding-ratio 0.2 --require-count 500 --max-images 500
```

Nếu bạn dùng file CSV export từ browser ở chỗ khác:

```powershell
python scripts/normalize_selected.py --selection-csv "D:\Downloads\reviewed_candidates.csv" --dataset-root data/raw/cub2002011 --padding-ratio 0.2 --require-count 500 --max-images 500
```

Output:

- `data/processed/images` (ảnh chuẩn hóa 224x224).
- `data/processed/metadata/images.csv`
- `data/processed/metadata/images.jsonl`
- `outputs/intermediate_examples` (kết quả trung gian cho báo cáo).

### Step 2.4 - Validation checklist

- Đúng `500` ảnh được chọn (`--require-count 500`).
- Tất cả ảnh đầu ra có kích thước `224x224`.
- Metadata có thông tin crop + resize.
- Có đủ ví dụ before/after trong `outputs/intermediate_examples`.

## Phase 3 - Re-run and Storage Management

- Có thể chạy lại từ Step 2.1 để tạo batch review mới.
- Nếu muốn làm sạch output cũ trước khi chạy lại:
  - xóa `outputs/review`
  - xóa `data/processed`
- Việc chạy lại nhiều lần không làm phình dung lượng nếu dọn output cũ định kỳ.
