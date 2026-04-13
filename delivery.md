# Báo Cáo Bàn Giao

## Phạm Vi Công Việc

Bản bàn giao này bao gồm:

1. Đánh giá đa tác tử đối với trạng thái hiện tại của repository dựa trên:
   - yêu cầu đề bài
   - góp ý của giảng viên
   - các tài liệu hiện tại [README.md](/d:/PTIT/Multimedia%20Database/README.md), [plan.md](/d:/PTIT/Multimedia%20Database/plan.md), và [roadmap.md](/d:/PTIT/Multimedia%20Database/roadmap.md)
2. Một số chỉnh sửa nhỏ nhưng cần thiết để dự án bám đúng định hướng Multimedia Database CBIR
3. Chạy lại sạch từ ranh giới Phase 2 đã review xong đến hết Phase 3 trích xuất đặc trưng

Bản bàn giao này **chủ động dừng ở bước trích xuất đặc trưng**. Phase 4 build lại database được để làm bước tiếp theo.

## Tóm Tắt Mức Độ Phù Hợp Với Đề Bài

Hướng hiện tại của dự án là đúng.

Đề bài yêu cầu:
- xây dựng bộ dữ liệu ảnh chim đã được thu thập và chuẩn hóa
- thiết kế bộ đặc trưng thể hiện cả sự tương đồng và khác biệt
- xây dựng hệ cơ sở dữ liệu để quản lý metadata và descriptor
- truy hồi `top-5` ảnh tương tự cho ảnh query
- trình bày sơ đồ khối, quy trình xử lý, kết quả trung gian, demo và đánh giá

Phản hồi của giảng viên nhấn mạnh:
- đây là bài về **Hệ CSDL Đa Phương Tiện**, không phải bài phân loại bằng machine learning
- báo cáo phải giải thích rõ:
  - dùng những đặc trưng nào
  - mỗi đặc trưng mang ý nghĩa thông tin gì
  - thuật toán trích rút đặc trưng là gì
  - cơ sở dữ liệu được tổ chức như thế nào
  - cơ chế tìm kiếm ảnh tương đồng hoạt động ra sao
  - hệ thống gồm những module nào, input/output của từng module là gì
  - các công thức, mô hình toán học nào được sử dụng
- đầu ra cần đúng theo **nội dung tương đồng**, không bắt buộc đúng cùng loài

Định hướng hiện tại của repo sau khi rà soát:
- DB-first: đúng
- loose coupling: đúng
- handcrafted features là phần lõi dễ giải thích: đúng
- CNN chỉ đóng vai trò phụ để so sánh/fusion: đúng
- retrieval theo similarity, không phải classification: đúng

## Đánh Giá Bằng Multi-Agent

Hai subagent chuyên biệt đã được sử dụng trong lần đánh giá này:

- tác tử `data-engineer`
  Agent id: `019d8776-c915-7ba2-98ab-e1c882041002`
- tác tử `database-optimizer`
  Agent id: `019d8776-e4d6-7723-bcbe-e4f323c8629d`

### Kết Luận Từ Nhánh Data Engineer

Pipeline dữ liệu hiện tại đúng về cấu trúc, nhưng có ba lỗ hổng thực dụng:

1. Phase 2 trước đó chưa an toàn khi rerun.
   - Ảnh processed cũ và metadata cũ có thể còn sót lại trên đĩa.
2. Artifact của Phase 3/4 trên đĩa đã cũ so với code hiện tại.
   - Feature và DB trước đó được build trước khi bổ sung logic foreground-aware extraction.
3. Export manual relevance trước đó chưa an toàn khi replay.
   - Retrieval run cũ có thể làm nhiễu template gán nhãn relevance.

Khuyến nghị của nhánh này:
- giữ lại các CSV review đã chốt
- dọn sạch từ boundary processed trở đi
- chạy lại normalization
- chạy lại extraction

### Kết Luận Từ Nhánh Database Optimizer

Schema SQLite hiện tại là phù hợp với bài tập lớn này.

Những điểm đang ổn:
- ảnh vẫn lưu trên filesystem
- SQLite quản lý metadata, feature vectors, query cache, judgments và experiment summaries
- retrieval thực hiện bằng cách đọc descriptor từ SQLite rồi chấm điểm trong application code
- cách này phù hợp với quy mô `1000` ảnh và dễ bảo vệ về mặt học thuật

Điểm cần chỉnh:
- mô tả “DB-backed retrieval” phải chính xác hơn
- không nên mô tả như thể SQLite đang trực tiếp thực hiện vector similarity search trong SQL

Khuyến nghị nhỏ ở mức schema/index:
- thêm composite index cho đường đánh giá latest run

Những thứ không nên over-engineer lúc này:
- ANN
- vector DB
- SQL-native similarity search
- thay SQLite bằng hệ khác
- refactor schema lớn

## Đánh Giá Tài Liệu Tham Khảo

Các file trong [documents](/d:/PTIT/Multimedia%20Database/documents) củng cố đúng hướng hiện tại.

Kết luận thực dụng từ tài liệu:
- hai tài liệu/sách về multimedia database management systems ủng hộ đúng cách trình bày hiện tại:
  - multimedia information retrieval system
  - feature extraction và content representation
  - metadata management và storage management
  - similarity-based retrieval
- file báo cáo PTIT mẫu chủ yếu hữu ích ở mặt bố cục báo cáo, không nên dùng làm định hướng kỹ thuật
  - nó nghiêng về classification
  - không nên lấy nó làm phương pháp luận chính cho repo này

Kết luận:
- giữ nguyên hướng CBIR + Multimedia Database hiện tại
- không quay lại hướng classification

## Các Chỉnh Sửa Đã Áp Dụng

### 1. Phase 2 normalization đã hỗ trợ clean rerun

Đã cập nhật:
- [scripts/phase2_normalize/normalize_selected.py](/d:/PTIT/Multimedia%20Database/scripts/phase2_normalize/normalize_selected.py)

Đã thêm:
- `--clean-output`

Tác dụng:
- trước khi sinh lại bộ processed cuối, script sẽ xóa:
  - `data/processed/images`
  - `data/processed/metadata`
  - `outputs/intermediate_examples`

Việc này xử lý lỗ hổng idempotency chính của Phase 2.

### 2. Export manual relevance giờ chỉ lấy latest runs

Đã cập nhật:
- [scripts/phase6_experiments/prepare_manual_relevance.py](/d:/PTIT/Multimedia%20Database/scripts/phase6_experiments/prepare_manual_relevance.py)

Tác dụng:
- template export cho manual relevance giờ chỉ dùng run mới nhất cho mỗi cặp `(query_id, experiment)`
- run cũ không còn làm bẩn template annotate ở các lần chạy sau

### 3. SQLite schema được bổ sung index nhỏ, ít rủi ro

Đã cập nhật:
- [scripts/shared/db_utils.py](/d:/PTIT/Multimedia%20Database/scripts/shared/db_utils.py)

Đã thêm:
- `idx_retrieval_runs_latest(query_id, experiment_id, run_id DESC)`
- `idx_relevance_judgments_source_query(judgment_source, query_id, image_id)`

Đây là các tối ưu nhỏ, không làm thay đổi kiến trúc tổng thể.

### 4. Phần mô tả cleanup đã được sửa cho đúng

Đã cập nhật:
- [scripts/phase3_descriptor_extraction/clean_phase_outputs.py](/d:/PTIT/Multimedia%20Database/scripts/phase3_descriptor_extraction/clean_phase_outputs.py)

Tác dụng:
- script giờ mô tả chính xác là đang xóa **artifact downstream từ Phase 3 trở đi**, không chỉ riêng Phase 3

### 5. Tài liệu đã được chỉnh lại cho chính xác hơn

Đã cập nhật:
- [README.md](/d:/PTIT/Multimedia%20Database/README.md)
- [plan.md](/d:/PTIT/Multimedia%20Database/plan.md)
- [roadmap.md](/d:/PTIT/Multimedia%20Database/roadmap.md)

Các điểm được làm rõ:
- retrieval đọc descriptor matrices từ SQLite
- similarity scoring được thực hiện trong application code
- UI demo là read-only, không persist query logs
- `images.width` và `images.height` là kích thước ảnh gốc
- kích thước normalized nằm trong preprocessing metadata

## Các Lệnh Đã Chạy

### Chạy lại Phase 2

```powershell
.\.venv\Scripts\python.exe scripts\phase2_normalize\normalize_selected.py `
  --selection-csv outputs/review/final_reviewed_candidates_1000.csv `
  --dataset-root data/raw/cub2002011 `
  --padding-ratio 0.35 `
  --clean-output `
  --require-count 1000 `
  --max-images 1000
```

### Dọn Phase 3+ downstream cũ

```powershell
.\.venv\Scripts\python.exe scripts\phase3_descriptor_extraction\clean_phase_outputs.py --yes
```

### Chạy lại Phase 3

```powershell
.\.venv\Scripts\python.exe scripts\phase3_descriptor_extraction\extract_features.py `
  --metadata-csv data/processed/metadata/images.csv `
  --processed-root data/processed `
  --output-dir data/features `
  --device auto
```

## Kết Quả Thực Thi

### Đầu ra của Phase 2

Đã kiểm tra sau khi rerun:
- số ảnh processed: `1000`
- số dòng metadata CSV: `1000`
- số file intermediate examples: `15`

Các output hiện tại:
- [data/processed/images](/d:/PTIT/Multimedia%20Database/data/processed/images)
- [data/processed/metadata/images.csv](/d:/PTIT/Multimedia%20Database/data/processed/metadata/images.csv)
- [data/processed/metadata/images.jsonl](/d:/PTIT/Multimedia%20Database/data/processed/metadata/images.jsonl)
- [outputs/intermediate_examples](/d:/PTIT/Multimedia%20Database/outputs/intermediate_examples)

### Đầu ra của Phase 3

Đã kiểm tra sau khi rerun:
- số dòng manifest: `1000`
- số dòng feature JSONL: `1000`
- toàn bộ descriptor đã được extract thành công cho `1000` ảnh

Các output hiện tại:
- [data/features/images_manifest.csv](/d:/PTIT/Multimedia%20Database/data/features/images_manifest.csv)
- [data/features/descriptor_table.csv](/d:/PTIT/Multimedia%20Database/data/features/descriptor_table.csv)
- [data/features/global_hsv_hist.npy](/d:/PTIT/Multimedia%20Database/data/features/global_hsv_hist.npy)
- [data/features/regional_hsv_hist.npy](/d:/PTIT/Multimedia%20Database/data/features/regional_hsv_hist.npy)
- [data/features/color_moments.npy](/d:/PTIT/Multimedia%20Database/data/features/color_moments.npy)
- [data/features/lbp_hist.npy](/d:/PTIT/Multimedia%20Database/data/features/lbp_hist.npy)
- [data/features/hog_descriptor.npy](/d:/PTIT/Multimedia%20Database/data/features/hog_descriptor.npy)
- [data/features/cnn_embedding.npy](/d:/PTIT/Multimedia%20Database/data/features/cnn_embedding.npy)
- [data/features/features.jsonl](/d:/PTIT/Multimedia%20Database/data/features/features.jsonl)
- [data/features/config.json](/d:/PTIT/Multimedia%20Database/data/features/config.json)

### Kiểm tra cấu hình Phase 3

Rút ra từ [config.json](/d:/PTIT/Multimedia%20Database/data/features/config.json):

- `dataset_version = normalized_gallery_1000`
- `cnn_backbone = resnet18`
- `device_used = cpu`
- `include_cnn = true`
- `foreground_focus = true`
- `regional_grid = [2, 2]`
- `hsv_bins = [8, 8, 8]`

Thống kê foreground-mask:
- `estimated_foreground_mask = 998`
- `bbox_rect_fallback = 2`

Kích thước các descriptor:
- `global_hsv_hist = 512`
- `regional_hsv_hist = 2048`
- `color_moments = 9`
- `lbp_hist = 10`
- `hog_descriptor = 6084`
- `cnn_embedding = 512`

## Đánh Giá Cuối Cùng

Trạng thái hiện tại của dự án đã sạch và chặt chẽ hơn trước lần rerun này.

Những điều hiện đã đúng:
- narrative của repo bám sát đề bài và góp ý của giảng viên
- bộ `1000` ảnh cuối đã được build lại sạch
- đặc trưng đã được trích xuất lại sạch
- artifact trên đĩa giờ khớp với code foreground-aware hiện tại và khớp với mô tả trong tài liệu

Những gì **chưa** được làm ở lần này:
- build lại SQLite ở Phase 4
- rerun retrieval ở Phase 5
- chạy experiments ở Phase 6
- đánh giá ở Phase 7
- export report artifacts ở Phase 8

Điểm dừng này là chủ động, vì milestone bạn yêu cầu là: **chạy lại đến hết bước trích xuất đặc trưng**.

## Bước Tiếp Theo Được Khuyến Nghị

Chạy Phase 4 ngay sau đó:

```powershell
.\.venv\Scripts\python.exe scripts\phase4_feature_database\build_feature_db.py `
  --features-dir data/features `
  --db-path data/features/cbir_features.sqlite `
  --overwrite
```

Sau đó tiếp tục:
1. Phase 5 retrieval
2. Phase 6 experiments và manual relevance
3. Phase 7 evaluation
4. Phase 8 report artifacts
