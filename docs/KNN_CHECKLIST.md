# Người 2 - Checklist KNN

Branch làm việc: `feature/knn`

## Mục tiêu
Hoàn thiện 2 thí nghiệm chính:
1. KNN + CountVectorizer
2. KNN + TF-IDF

và tuning K trên `train.csv`.

## Việc cần làm

### A. Chuẩn bị
- [ ] Pull/cập nhật code mới nhất từ `develop`.
- [ ] Chạy `pip install -r requirements.txt`.
- [ ] Chạy `python scripts/download_dataset.py`.
- [ ] Xác nhận `data/raw/train.csv` và `data/raw/test.csv` tồn tại.
- [ ] Chạy test chung: `python -m pytest tests`.

### B. Kiểm tra dữ liệu
- [ ] In số dòng train/test.
- [ ] Kiểm tra `message`, `label`.
- [ ] Kiểm tra missing/duplicate.
- [ ] Thống kê Ham/Spam.
- [ ] Đối chiếu thống kê với Người 1.

### C. Preprocessing
- [ ] Chỉ dùng `common.preprocessing.clean_series`.
- [ ] Kiểm tra 10 tin nhắn mẫu trước/sau preprocessing.
- [ ] Không tự xóa số/token ẩn danh.
- [ ] Không dùng preprocessing riêng.

### D. Vectorization
- [ ] CountVectorizer với config chung.
- [ ] TF-IDF với config chung.
- [ ] Fit trên train, transform test.
- [ ] Ghi shape ma trận.

### E. Tuning K
- [ ] Thử `K = 3, 5, 7, 9, 11, 15, 21`.
- [ ] Tuning chỉ trên `train.csv`.
- [ ] Không nhìn `test.csv` để chọn K.
- [ ] Tính F1 cho từng K.
- [ ] Lưu bảng K -> F1.
- [ ] Chọn best K riêng cho Count và TF-IDF.
- [ ] Vẽ biểu đồ K vs F1.

### F. Train cuối
- [ ] Train KNN + Count bằng best K.
- [ ] Train KNN + TF-IDF bằng best K.
- [ ] Đo training time.
- [ ] Đo prediction time.

### G. Evaluation
- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1-score.
- [ ] Confusion Matrix cho 2 cấu hình.
- [ ] 5 False Positive.
- [ ] 5 False Negative.
- [ ] Phân tích lỗi.

### H. Output bắt buộc
- [ ] `results/knn/results_knn.csv`.
- [ ] Bảng tuning K.
- [ ] Biểu đồ K vs F1.
- [ ] Confusion matrix Count.
- [ ] Confusion matrix TF-IDF.
- [ ] Notebook giải thích thí nghiệm.
- [ ] Nội dung báo cáo phần KNN.

## Điều kiện hoàn thành
- `python scripts/run_knn.py` chạy không lỗi.
- Best K được chọn từ train/validation, không từ test.
- Có đủ Count + TF-IDF.
- Có error analysis và runtime comparison.
- Không sửa code Naive Bayes.
