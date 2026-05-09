# Yêu Cầu Và Tiêu Chí Bàn Giao

Tài liệu này diễn giải yêu cầu ban đầu của giảng viên thành các tiêu chí kỹ thuật cần có trong dự án **Bird Image Storage and Retrieval**. Trạng thái artifact hiện tại được theo dõi riêng trong `docs/current_status.md`.

## Mục Tiêu Bài Tập

Thiết kế và triển khai một hệ thống cơ sở dữ liệu đa phương tiện để **lưu trữ ảnh chim và truy xuất các ảnh chim tương tự theo nội dung hình ảnh**.

Hệ thống không được trình bày như một bài toán phân loại loài chim. Đầu vào là ảnh chim truy vấn, đầu ra là `top-5` ảnh giống nhất trong gallery, được xếp hạng giảm dần theo độ tương đồng nội dung.

## 1. Dataset Ảnh Chim

Yêu cầu gốc:

- Có ít nhất `500` file ảnh chim.
- Tất cả ảnh có cùng kích thước.
- Đối tượng chim trong ảnh có cùng tỉ lệ khung hình.
- Chim phải ở tư thế đậu, không bay.
- Góc nhìn phải nhất quán theo hướng ngang.
- Sinh viên được tự chọn định dạng ảnh.

Tiêu chí thiết kế trong dự án:

- Sử dụng CUB-200-2011 làm nguồn dữ liệu chính.
- Lọc ảnh theo tiêu chí perching, side-view/horizontal view, rõ đối tượng, ít che khuất.
- Chuẩn hóa ảnh về cùng kích thước bằng ROI crop theo bounding box và resize.
- Lưu trace tiền xử lý: bounding box, crop box, kích thước nguồn, kích thước sau chuẩn hóa, cờ `keep`, ghi chú review.

Bằng chứng cần có trong báo cáo:

- Mô tả nguồn dữ liệu.
- Quy trình lọc và chuẩn hóa.
- Ví dụ ảnh trước/sau normalization.
- Bảng metadata chính của gallery.

## 2. Feature Design

Yêu cầu gốc:

- Thiết kế tập feature dùng để nhận diện/truy xuất ảnh chim trong dataset.
- Feature phải bao gồm cả thông tin thể hiện sự giống nhau và thông tin thể hiện sự khác nhau giữa các ảnh chim.
- Phải giải thích lý do chọn feature và phân tích giá trị thông tin của từng feature.

Feature chính của dự án là handcrafted descriptors:

| Feature | Vai trò | Giá trị thông tin |
| --- | --- | --- |
| `global_hsv_hist` | Similarity | Mô tả phân bố màu tổng thể của chim. |
| `regional_hsv_hist` | Similarity + difference | Mô tả bố cục màu theo vùng, giúp phân biệt vị trí màu trên thân/đầu/cánh. |
| `color_moments` | Similarity | Mô tả thống kê màu gọn nhẹ qua mean, std, skewness. |
| `lbp_hist` | Difference | Mô tả texture cục bộ, hữu ích cho hoa văn lông. |
| `hog_descriptor` | Difference | Mô tả cạnh, contour, pose, và cấu trúc hình học. |
| `silhouette_shape_descriptor` | Similarity + difference | Mô tả dáng tổng thể và hình học foreground. |
| `cnn_embedding` | Baseline phụ trợ | Mô tả ngữ nghĩa thị giác bằng ResNet18 pretrained; không dùng như classifier. |

Ràng buộc báo cáo:

- Handcrafted descriptors là phần chính để giải thích.
- CNN chỉ là baseline hoặc tín hiệu phụ trong fusion.
- Không dùng `species_id` để lọc hoặc boost kết quả retrieval.
- Không trình bày kết quả chính bằng classification accuracy.

## 3. Database System

Yêu cầu gốc:

- Phát triển hệ thống cơ sở dữ liệu để quản lý metadata sẵn có.
- Mô tả cơ chế truy xuất ảnh chim tương tự dựa trên metadata và feature đã lưu.

Thiết kế trong dự án:

- Kiến trúc loose coupling:
  - file ảnh nằm trên filesystem
  - metadata và descriptors nằm trong SQLite
- SQLite là system-of-record cho:
  - `images`
  - `preprocessing_metadata`
  - `feature_types`
  - `image_features`
  - `queries`
  - `query_features`
  - `experiments`
  - `retrieval_runs`
  - `retrieval_results`
  - `relevance_judgments`

Cơ chế retrieval:

1. Query image được normalize và trích xuất descriptor.
2. Retrieval engine đọc descriptor gallery từ SQLite.
3. Tính similarity giữa query vector và từng gallery vector.
4. Calibrate score nếu experiment yêu cầu.
5. Weighted fusion các descriptor score.
6. Sort giảm dần theo fused score.
7. Trả về `top-5` ảnh giống nhất.

Điểm cần nhấn mạnh trong báo cáo:

- SQLite không chạy native vector search.
- Similarity scan và fusion chạy ở application layer.
- Với quy mô bài tập, exhaustive kNN/linear scan là baseline minh bạch và phù hợp.

## 4. Retrieval System

Yêu cầu gốc:

- Input là một ảnh chim mới.
- Ảnh truy vấn có thể thuộc loài đã có trong dataset hoặc loài chưa từng thấy.
- Output là `5` ảnh giống nhất, xếp hạng giảm dần theo content similarity.
- Phải trình bày block diagram và workflow xử lý.
- Phải trình bày intermediate results của quá trình retrieval.

Các mode demo chính:

- `calibrated_handcrafted`: mode chính cho báo cáo, chỉ dùng handcrafted descriptors đã calibrate score.
- `fusion`: handcrafted-first, thêm CNN embedding với vai trò phụ trợ.
- `cnn_only`: baseline tham khảo.

Intermediate results nên trình bày:

- ảnh query sau normalization
- foreground mask hoặc crop/bbox minh họa
- descriptor table
- per-feature score breakdown
- weighted contribution trong fusion
- top-5 ranked results

## 5. Demonstration And Evaluation

Yêu cầu gốc:

- Demonstrate hệ thống.
- Đánh giá kết quả đạt được.

Demo:

- Sử dụng Gradio UI.
- Upload ảnh chim truy vấn.
- Chọn một trong ba mode demo chính.
- Giữ `Top K = 5`.
- Trình bày ảnh kết quả và JSON payload để giải thích điểm số.

Evaluation:

- Metric chính: manual visual relevance với `nDCG@5` và `Precision@5`.
- Relevance nên được chấm theo mức độ giống thị giác, không chỉ theo cùng loài.
- Species-proxy chỉ là baseline phụ vì nó gần bài toán classification.
- Auto-visual-proxy chỉ dùng để sanity-check hoặc hỗ trợ review, không dùng làm ground truth cuối.

## Checklist Báo Cáo Cuối

- Dataset source và tiêu chí lọc ảnh.
- Quy trình normalization để đảm bảo cùng kích thước, cùng tỉ lệ đối tượng, cùng góc nhìn.
- Bảng feature và giải thích information value.
- Schema/ERD của SQLite database.
- System block diagram.
- Workflow retrieval end-to-end.
- Công thức similarity: chi-square, inverse Euclidean, cosine.
- Công thức weighted fusion và score calibration.
- Ví dụ top-5 retrieval kèm score breakdown.
- Đánh giá bằng manual `nDCG@5` và `Precision@5`.
- Nhận xét giới hạn và hướng mở rộng.
