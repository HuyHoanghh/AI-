# Người 1 - Checklist Naive Bayes

Branch làm việc: `feature/naive-bayes`

## Mục tiêu
Hoàn thiện 2 thí nghiệm:
1. Multinomial Naive Bayes + CountVectorizer
2. Multinomial Naive Bayes + TF-IDF

## Việc cần làm

### A. Chuẩn bị
- [ ] Pull/cập nhật code mới nhất từ `develop`.
- [ ] Chạy `pip install -r requirements.txt`.
- [ ] Chạy `python scripts/download_dataset.py`.
- [ ] Xác nhận `data/raw/train.csv` và `data/raw/test.csv` tồn tại.
- [ ] Chạy test chung: `python -m pytest tests`.

### B. Kiểm tra dữ liệu
- [ ] In số dòng train/test.
- [ ] Kiểm tra cột `message`, `label`.
- [ ] Kiểm tra missing value.
- [ ] Kiểm tra duplicate.
- [ ] Thống kê số Ham/Spam.
- [ ] Ghi kết quả để đưa vào báo cáo.

### C. EDA
- [ ] Phân bố nhãn Ham/Spam.
- [ ] Độ dài ký tự và số từ của tin nhắn.
- [ ] Top từ phổ biến ở Ham.
- [ ] Top từ phổ biến ở Spam.
- [ ] Thống kê token `[PHONE]`, `[MONEY]`, `[NUMBER]`, `[TIME]`, `[DATE]`, `[BANK_ACC]`.
- [ ] Lưu biểu đồ cần dùng cho báo cáo.

### D. Preprocessing
- [ ] Chỉ dùng `common.preprocessing.clean_series`.
- [ ] Kiểm tra 10 tin nhắn mẫu trước/sau preprocessing.
- [ ] Không xóa token ẩn danh.
- [ ] Không tự viết preprocessing khác trong notebook.

### E. Vectorization
- [ ] CountVectorizer với config chung.
- [ ] TF-IDF với config chung.
- [ ] Fit vectorizer trên train.
- [ ] Chỉ transform test.
- [ ] Ghi shape ma trận train/test.

### F. Naive Bayes
- [ ] Train NB + Count.
- [ ] Train NB + TF-IDF.
- [ ] Đo training time.
- [ ] Đo prediction time.

### G. Evaluation
- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] F1-score.
- [ ] Confusion Matrix cho 2 cấu hình.
- [ ] Lấy ít nhất 5 False Positive.
- [ ] Lấy ít nhất 5 False Negative.
- [ ] Phân tích nguyên nhân sai.

### H. Cross-validation
- [ ] Chạy 5-fold CV trên `train.csv` nếu đủ thời gian.
- [ ] Không đưa `test.csv` vào CV.
- [ ] Ghi Mean F1 và Std F1.

### I. Output bắt buộc
- [ ] `results/naive_bayes/results_nb.csv`.
- [ ] Confusion matrix Count.
- [ ] Confusion matrix TF-IDF.
- [ ] Notebook giải thích thí nghiệm.
- [ ] Nội dung báo cáo phần Naive Bayes.

## Điều kiện hoàn thành
- `python scripts/run_nb.py` chạy không lỗi.
- Có đủ 2 dòng Count/TF-IDF trong kết quả.
- Không có data leakage.
- Có error analysis.
- Không sửa KNN branch/code.
