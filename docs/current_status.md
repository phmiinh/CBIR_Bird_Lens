# Rà Soát Trạng Thái Hiện Tại

Cập nhật: `2026-05-08` theo múi giờ Asia/Saigon

File này trả lời trực tiếp các vấn đề trong phần đánh giá cũ, dựa trên trạng thái artifact hiện tại của workspace và kết quả rà soát multi-agent mới.

## Kết Luận Nhanh

Đánh giá cũ không còn đúng hoàn toàn với workspace hiện tại.

Trạng thái đã xác nhận:

- Phase 1 có dữ liệu raw CUB.
- Phase 2 đã có gallery chuẩn hóa `1000` ảnh.
- Phase 3 đã extract đủ `7` descriptors.
- `silhouette_shape_descriptor` đã được implement, extract, lưu ra `.npy`, và nạp vào SQLite.
- Phase 4 build SQLite DB đã hoàn tất.
- Phase 5/6 đã chạy retrieval experiments cho `50` queries x `6` experiments x `top-5`.
- Phase 7 mới có species-proxy evaluation.
- Phase 8 đã export report artifacts trong lần rà soát này.
- Manual relevance annotation vẫn là phần thiếu quan trọng nhất trước khi viết kết quả đánh giá cuối.

## Rà Soát Multi-Agent

Hai local agent trong `.codex/agents` đã được dùng:

- `data-engineer`
  - agent id: `019e0835-2338-7843-b19d-7dde2d9b945a`
  - phạm vi: source-to-sink pipeline, feature artifacts, descriptor dimensions, fusion/evaluation readiness
- `database-optimizer`
  - agent id: `019e0835-4fda-78c3-bbd1-e10797a13a52`
  - phạm vi: SQLite schema, vector storage, retrieval access pattern, weight normalization, performance tradeoff

Kết luận chung của hai agent:

- artifact hiện tại khớp với kiến trúc DB-first
- `silhouette_shape_descriptor` không còn thiếu
- DB và retrieval/evaluation species-proxy đã tồn tại
- manual relevance vẫn chưa được gán nhãn

## Rà Soát Theo Đánh Giá Mới

| Vấn đề trong đánh giá | Trạng thái sau rà soát | Việc đã làm hoặc cần làm |
| --- | --- | --- |
| `lbp_hist` chỉ có `10` chiều | Code đúng: uniform LBP dùng `P=8`, `R=1`, nên histogram có `P + 2 = 10` bins. | Đã bổ sung công thức vào `docs/roadmap.md`, và report descriptor export. |
| `hog_descriptor` có `6084` chiều | Code đúng: `224x224`, `pixels_per_cell=(16,16)`, `cells_per_block=(2,2)`, `orientations=9` -> `13 x 13 x 4 x 9 = 6084`. | Đã bổ sung công thức vào tài liệu và report descriptor export. |
| Fusion có thể bị tổng weight `1.20` | Hiện tại không còn bug này: code và DB đều có tổng weight `1.00`; `fusion = 0.80` handcrafted + `0.20` CNN. | Đã thêm runtime guard để fail nếu experiment weight không sum về `1.00`. |
| `vector_json` cần nêu tradeoff | Thiết kế phù hợp quy mô môn học vì minh bạch và reproducible, nhưng tốn storage/parse time. DB hiện khoảng `89.96 MB`, HOG JSON chiếm khoảng `49.6 MB`. | Đã thêm đoạn tradeoff trong `docs/roadmap.md`; nếu scale lớn thì chuyển vector sang BLOB/sidecar/vector index. |
| DB có thật sự retrieval không | Có, nhưng retrieval logic nằm ở application layer: SQLite là system-of-record cho descriptors, còn exhaustive kNN/fusion chạy trong Python. | Đã làm rõ trong `docs/roadmap.md`, và `system_block_diagram.md` export. |
| Manual relevance | Vẫn là blocker chính. SQLite hiện chỉ có `species_proxy`; template manual còn trống. | Cần điền label, import, rerun Phase 7/8 trước khi viết kết quả cuối. |

Runtime hardening đã bổ sung:

- `RetrievalEngine` lazy-load feature matrices theo experiment thay vì parse toàn bộ `7` descriptors ngay khi khởi tạo.
- `_load_experiment` kiểm tra tổng weight phải bằng `1.00` và feature phải tồn tại trong DB.
- `load_gallery_feature_matrix` kiểm tra thiếu/trùng vector và trả matrix đúng theo thứ tự `image_ids`.

## Trả Lời Các Vấn Đề Nghiêm Trọng

### 1. `silhouette_shape_descriptor` không còn thiếu

Nhận xét cũ đúng với `docs/delivery.md` cũ, nhưng không đúng với workspace hiện tại.

Bằng chứng:

- file tồn tại: `data/features/silhouette_shape_descriptor.npy`
- shape: `(1000, 15)`
- dtype: `float32`
- có trong `data/features/config.json`
- có trong `data/features/descriptor_table.csv`
- có trong SQLite table `feature_types`
- có `1000` rows trong SQLite table `image_features`
- có `50` query-feature rows trong SQLite table `query_features`

Kích thước descriptor hiện tại:

| Descriptor | Dimension | Trạng thái |
| --- | ---: | --- |
| `global_hsv_hist` | `512` | hoàn tất |
| `regional_hsv_hist` | `2048` | hoàn tất |
| `color_moments` | `9` | hoàn tất |
| `lbp_hist` | `10` | hoàn tất |
| `hog_descriptor` | `6084` | hoàn tất |
| `silhouette_shape_descriptor` | `15` | hoàn tất |
| `cnn_embedding` | `512` | hoàn tất |

### 2. Phase 4 đến Phase 8 không còn đều thiếu

Trạng thái hiện tại:

| Phase | Trạng thái hiện tại | Bằng chứng |
| --- | --- | --- |
| Phase 1 - Dataset setup | hoàn tất | `data/raw/cub2002011` |
| Phase 2 - Normalization | hoàn tất | `1000` processed images, `1000` metadata rows |
| Phase 3 - Feature extraction | hoàn tất | `7` descriptor matrices, mỗi descriptor có `1000` rows |
| Phase 4 - SQLite DB | hoàn tất | `data/features/cbir_features.sqlite` |
| Phase 5 - Retrieval | đã kiểm chứng kỹ thuật | CLI smoke output và persisted retrieval runs |
| Phase 6 - Experiments | hoàn tất phần chạy, chưa hoàn tất manual labels | `50` queries, `300` runs, `1500` top-k rows |
| Phase 7 - Evaluation | một phần | có species-proxy outputs, thiếu manual outputs |
| Phase 8 - Report artifacts | hoàn tất trong lần rà soát này | `outputs/report_artifacts/*` |

SQLite row counts:

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
| `relevance_judgments` | `50000` species-proxy rows |

### 3. Manual relevance annotation vẫn chưa hoàn tất

Đây vẫn là blocker chính trước khi viết kết quả đánh giá cuối.

Bằng chứng:

- `outputs/manual_relevance/manual_relevance_template.csv` tồn tại
- file này có `774` dòng query-candidate cần gán nhãn
- toàn bộ `relevance_grade` đang trống
- SQLite hiện chỉ có `judgment_source = 'species_proxy'`
- chưa có `outputs/eval/manual_per_query.csv`
- chưa có `outputs/eval/manual_summary.json`

Vì sao là `774` dòng, không phải chỉ `250` dòng:

- hiện có kết quả từ tất cả experiment configurations
- volume gốc là `50 queries x 6 experiments x 5 = 1500` rows
- template đã deduplicate các cặp query-candidate trùng nhau, còn `774` rows cần annotate

Nếu báo cáo chỉ muốn đánh giá một cấu hình cuối cùng, có thể scope annotation hẹp hơn. Nhưng template hiện tại là cơ sở đầy đủ hơn để so sánh mọi experiment/ablation.

## Kết Quả Evaluation Hiện Có

Species-proxy `Precision@5` hiện có:

| Experiment | Species-proxy Precision@5 |
| --- | ---: |
| `handcrafted_only` | `0.040` |
| `cnn_only` | `0.232` |
| `fusion` | `0.052` |
| `ablation_no_regional_color` | `0.028` |
| `ablation_no_explicit_shape` | `0.032` |
| `ablation_no_shape` | `0.028` |

Các số này chỉ nên trình bày như baseline phụ. Không nên dùng species-proxy để kết luận chất lượng CBIR cuối cùng, vì bài này là retrieval theo độ tương đồng thị giác chứ không phải species classification.

## Report Artifacts Hiện Có

Phase 8 export đã được chạy trong lần rà soát này và sinh ra:

- `outputs/report_artifacts/descriptor_table.csv`
- `outputs/report_artifacts/system_block_diagram.md`
- `outputs/report_artifacts/experiment_summaries.csv`
- `outputs/report_artifacts/example_retrieval_breakdown.json`

Giới hạn hiện tại:

- `experiment_summaries.csv` mới có species-proxy metrics
- sau khi manual relevance được fill và import, cần chạy lại Phase 7 và Phase 8 để artifact có manual `nDCG@5` và manual `Precision@5`

## Những Việc Còn Lại Trước Khi Viết Báo Cáo Cuối

1. Điền `relevance_grade` trong `outputs/manual_relevance/manual_relevance_template.csv`.
2. Import labels bằng `scripts/phase6_experiments/import_relevance_judgments.py`.
3. Chạy lại `scripts/phase7_evaluation/evaluate_retrieval.py`.
4. Chạy lại `scripts/phase8_report_artifacts/export_report_artifacts.py`.
5. Dùng artifact đã refresh để viết phần kết quả trong báo cáo.

Báo cáo vẫn cần chuẩn bị thêm:

- bảng I/O từng module
- bảng ý nghĩa thông tin của từng descriptor
- công thức Chi-square, Euclidean, cosine similarity, weighted score fusion
- công thức `Precision@5` và `nDCG@5`
- ERD/schema diagram
- hình intermediate results
- top-5 retrieval example với per-feature score breakdown


