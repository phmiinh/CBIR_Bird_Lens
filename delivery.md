# Báo Cáo Bàn Giao Và Cập Nhật Trạng Thái

Cập nhật: `2026-05-08` theo múi giờ Asia/Saigon

File này thay thế nội dung bàn giao cũ đã lỗi thời. Bản cũ nói pipeline chủ động dừng ở Phase 3, nhưng workspace hiện tại đã có artifact cho Phase 4, Phase 5/6, Phase 7 species-proxy và Phase 8.

## Kết Luận Ngắn

Hướng kỹ thuật của repo vẫn đúng với yêu cầu môn Cơ sở dữ liệu đa phương tiện:

- hệ thống là CBIR / multimedia retrieval, không phải classifier
- ảnh được lưu trên filesystem, metadata và descriptors được quản lý trong SQLite
- retrieval là approximate similarity ranking bằng descriptor, không phải exact-match SQL
- handcrafted descriptors là phần chính để giải thích, CNN chỉ là baseline/phụ trợ

Trạng thái hiện tại:

- Phase 1: hoàn tất
- Phase 2: hoàn tất với `1000` ảnh normalized
- Phase 3: hoàn tất với đủ `7` descriptors
- Phase 4: hoàn tất với SQLite DB
- Phase 5/6: hoàn tất với persisted retrieval experiments
- Phase 7: một phần, mới có species-proxy evaluation
- Phase 8: hoàn tất, đã export report artifacts
- Manual relevance: chưa xong và là blocker chính trước khi viết kết quả đánh giá cuối

## Multi-Agent Review Mới

Hai subagent local trong `.codex/agents` đã được dùng để kiểm tra độc lập:

- `data-engineer`
  - agent id: `019e0835-2338-7843-b19d-7dde2d9b945a`
  - phạm vi: source-to-sink pipeline, feature artifacts, descriptor dimensions, fusion/evaluation readiness
- `database-optimizer`
  - agent id: `019e0835-4fda-78c3-bbd1-e10797a13a52`
  - phạm vi: SQLite schema, vector storage, retrieval access pattern, weight normalization, performance tradeoff

Kết quả chung:

- artifact hiện tại khớp với kiến trúc DB-first
- `silhouette_shape_descriptor` không còn thiếu
- DB và retrieval/evaluation species-proxy đã tồn tại
- manual relevance vẫn chưa được gán nhãn
- các điểm đánh giá mới về LBP/HOG, fusion normalization, `vector_json`, và app-layer retrieval đã được rà soát; code đúng, tài liệu đã được bổ sung để giải thích rõ hơn
- retrieval runtime đã được harden thêm: lazy-load feature matrices theo experiment, validate tổng weight `1.00`, và kiểm tra thiếu/trùng vector khi load gallery features

## Trạng Thái Artifact Hiện Tại

| Hạng mục | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Raw CUB data | hoàn tất | `data/raw/cub2002011` |
| Processed gallery | hoàn tất | `1000` ảnh trong `data/processed/images` |
| Metadata | hoàn tất | `data/processed/metadata/images.csv` có `1000` dòng |
| Feature matrices | hoàn tất | `7` file `.npy` trong `data/features` |
| SQLite DB | hoàn tất | `data/features/cbir_features.sqlite` |
| Retrieval experiments | hoàn tất | `outputs/experiments/retrieval_runs.csv` có `300` dòng |
| Manual template | đã export | `outputs/manual_relevance/manual_relevance_template.csv` |
| Species-proxy eval | hoàn tất | `outputs/eval/species_proxy_summary.json` |
| Manual eval | chưa xong | chưa có `manual_summary.json` |
| Report artifacts | hoàn tất | `outputs/report_artifacts/*` |

## Xác Nhận `silhouette_shape_descriptor`

Vấn đề cũ: tài liệu cũ nói thiếu `silhouette_shape_descriptor.npy`.

Trạng thái mới: đã có và đã được kiểm chứng.

- file: `data/features/silhouette_shape_descriptor.npy`
- shape: `(1000, 15)`
- dtype: `float32`
- có trong `data/features/config.json`
- có trong `data/features/descriptor_table.csv`
- có trong SQLite table `feature_types`
- có `1000` rows trong `image_features`
- có `50` query-feature rows trong `query_features`

Kích thước descriptor hiện tại:

- `global_hsv_hist = 512`
- `regional_hsv_hist = 2048`
- `color_moments = 9`
- `lbp_hist = 10`
- `hog_descriptor = 6084`
- `silhouette_shape_descriptor = 15`
- `cnn_embedding = 512`

## SQLite Status

SQLite DB hiện tại có các row counts:

| Table | Rows |
| --- | ---: |
| `images` | `1000` |
| `preprocessing_metadata` | `1000` |
| `feature_types` | `7` |
| `image_features` | `7000` |
| `queries` | `50` |
| `query_features` | `350` |
| `retrieval_runs` | `300` |
| `retrieval_results` | `1500` |
| `experiments` | `6` |
| `relevance_judgments` | `50000` |

`relevance_judgments` hiện chỉ có `judgment_source = 'species_proxy'`.

## Evaluation Status

Species-proxy evaluation đã có:

| Experiment | Species-proxy Precision@5 |
| --- | ---: |
| `handcrafted_only` | `0.040` |
| `cnn_only` | `0.232` |
| `fusion` | `0.052` |
| `ablation_no_regional_color` | `0.028` |
| `ablation_no_explicit_shape` | `0.032` |
| `ablation_no_shape` | `0.028` |

Cần ghi rõ trong báo cáo:

- species-proxy chỉ là baseline yếu
- không nên kết luận chất lượng CBIR dựa trên species-proxy
- metric chính vẫn phải là manual `nDCG@5`

## Manual Relevance Status

Manual relevance chưa xong.

Bằng chứng:

- `outputs/manual_relevance/manual_relevance_template.csv` tồn tại
- file này có `774` dòng cần annotate
- tất cả `relevance_grade` đang trống
- chưa có manual judgments trong SQLite
- chưa có `outputs/eval/manual_per_query.csv`
- chưa có `outputs/eval/manual_summary.json`

Lý do có `774` dòng:

- experiments hiện có `50 x 6 x 5 = 1500` top-k result rows
- template đã deduplicate các cặp query-candidate trùng nhau
- sau dedup còn `774` cặp cần chấm điểm

## Phase 8 Report Artifacts

Đã export Phase 8 trong lần cập nhật này:

- `outputs/report_artifacts/descriptor_table.csv`
- `outputs/report_artifacts/system_block_diagram.md`
- `outputs/report_artifacts/experiment_summaries.csv`
- `outputs/report_artifacts/example_retrieval_breakdown.json`

Giới hạn:

- `experiment_summaries.csv` hiện chỉ có species-proxy metrics
- sau khi import manual judgments, cần rerun Phase 7 và Phase 8

## Các Việc Còn Lại

Thứ tự nên làm tiếp:

1. Điền `relevance_grade` trong `outputs/manual_relevance/manual_relevance_template.csv`.
2. Import manual judgments bằng `scripts/phase6_experiments/import_relevance_judgments.py`.
3. Rerun Phase 7 để sinh manual `nDCG@5` và manual `Precision@5`.
4. Rerun Phase 8 để update report artifacts.
5. Viết báo cáo cuối dựa trên artifact đã refresh.

Nội dung báo cáo vẫn cần chuẩn bị thêm:

- bảng I/O từng module
- công thức Chi-square, Euclidean, cosine, weighted fusion
- công thức `Precision@5` và `nDCG@5`
- bảng ý nghĩa thông tin của từng descriptor
- ERD/schema diagram
- hình intermediate result
- top-5 retrieval example với per-feature score
