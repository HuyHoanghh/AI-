# Người 2 - Checklist KNN

Branch làm việc: `feature/knn`

## Mục tiêu
Hoàn thiện 2 thí nghiệm chính:
1. KNN + CountVectorizer
2. KNN + TF-IDF

và tuning K trên `train.csv`.

## Việc cần làm

### A. Chuẩn bị
- [x] Pull/cập nhật code mới nhất từ `develop`.
- [x] Chạy `pip install -r requirements.txt`.
- [x] Chạy `python scripts/download_dataset.py`.
- [x] Xác nhận `data/raw/train.csv` và `data/raw/test.csv` tồn tại.
- [x] Chạy test chung: `python -m pytest tests`.

### B. Kiểm tra dữ liệu
- [x] In số dòng train/test.
- [x] Kiểm tra `message`, `label`.
- [x] Kiểm tra missing/duplicate.
- [x] Thống kê Ham/Spam.
- [x] Đối chiếu thống kê với Người 1.

### C. Preprocessing
- [x] Chỉ dùng `common.preprocessing.clean_series`.
- [x] Kiểm tra 10 tin nhắn mẫu trước/sau preprocessing.
- [x] Không tự xóa số/token ẩn danh.
- [x] Không dùng preprocessing riêng.

### D. Vectorization
- [x] CountVectorizer với config chung.
- [x] TF-IDF với config chung.
- [x] Fit trên train, transform test.
- [x] Ghi shape ma trận.

### E. Tuning K
- [x] Thử `K = 3, 5, 7, 9, 11, 15, 21`.
- [x] Tuning chỉ trên `train.csv`.
- [x] Không nhìn `test.csv` để chọn K.
- [x] Tính F1 cho từng K.
- [x] Lưu bảng K -> F1.
- [x] Chọn best K riêng cho Count và TF-IDF.
- [x] Vẽ biểu đồ K vs F1.

### F. Train cuối
- [x] Train KNN + Count bằng best K.
- [x] Train KNN + TF-IDF bằng best K.
- [x] Đo training time.
- [x] Đo prediction time.

### G. Evaluation
- [x] Accuracy.
- [x] Precision.
- [x] Recall.
- [x] F1-score.
- [x] Confusion Matrix cho 2 cấu hình.
- [x] 5 False Positive.
- [x] 5 False Negative.
- [x] Phân tích lỗi.

### H. Output bắt buộc
- [x] `results/knn/results_knn.csv`.
- [x] Bảng tuning K.
- [x] Biểu đồ K vs F1.
- [x] Confusion matrix Count.
- [x] Confusion matrix TF-IDF.
- [x] Notebook giải thích thí nghiệm.
- [x] Nội dung báo cáo phần KNN.

## Điều kiện hoàn thành
- `python scripts/run_knn.py` chạy không lỗi.
- Best K được chọn từ train/validation, không từ test.
- Có đủ Count + TF-IDF.
- Có error analysis và runtime comparison.
- Không sửa code Naive Bayes.