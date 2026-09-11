# Checklist chung - Vietnamese SMS Spam Detection

Checklist này là chuẩn chung cho cả 2 thành viên. Hai nhánh model được làm song song nhưng phải tuân thủ cùng dữ liệu, preprocessing, vectorizer và metrics để kết quả cuối so sánh công bằng.

## 1. Chuẩn dữ liệu bắt buộc

- Dataset chính thức: `trannguyenthaituan251209/vietnamese_sms_dataset`.
- `train.csv`: dùng huấn luyện, tuning và cross-validation.
- `test.csv`: chỉ dùng đánh giá cuối cùng.
- Text column: `message`.
- Label column: `label`.
- `0 = Ham/Hợp lệ`, `1 = Spam/Lừa đảo`.
- Không dùng `test.csv` để chọn hyperparameter.
- Không fit vectorizer trên test set.

## 2. Chuẩn preprocessing

Cả hai model phải gọi `common.preprocessing.clean_series`.

Bắt buộc:
- xử lý missing thành chuỗi rỗng;
- chuẩn hóa Unicode NFC;
- lowercase;
- URL -> token `__url__`;
- email -> token `__email__`;
- bảo toàn các token ẩn danh của dataset: PHONE, BANK_ACC, MONEY, NUMBER, TIME, DATE;
- không xóa toàn bộ chữ số một cách máy móc;
- chuẩn hóa khoảng trắng.

Không tự viết pipeline preprocessing khác trong notebook/model riêng.

## 3. Chuẩn biểu diễn văn bản

Hai pipeline phải thử đúng 2 cấu hình:

1. CountVectorizer
2. TF-IDF

Thông số dùng chung lấy từ `config/settings.py`:
- `MAX_FEATURES = 5000`
- `NGRAM_RANGE = (1, 1)`

Vectorizer chỉ `fit` trên train và chỉ `transform` trên test.

## 4. Chuẩn đánh giá

Mỗi cấu hình phải báo cáo:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training time
- Prediction time

Ưu tiên F1-score khi so sánh model vì lớp Spam/Lừa đảo là lớp cần phát hiện.

## 5. File kết quả bắt buộc

Naive Bayes:
- `results/naive_bayes/results_nb.csv`

KNN:
- `results/knn/results_knn.csv`
- nên có thêm bảng tuning K.

Kết quả cuối:
- `results/final/final_comparison.csv`

## 6. Quy tắc Git

- Người 1 làm trên `feature/naive-bayes`.
- Người 2 làm trên `feature/knn`.
- Không push trực tiếp vào `main`.
- Không tự sửa `common/` hoặc `config/` trên feature branch nếu chưa thống nhất.
- Trước khi làm việc nên cập nhật từ `develop`.
- Hoàn thành phần việc -> Pull Request vào `develop`.
- Sau khi integration test thành công mới merge `develop -> main`.

## 7. Definition of Done

Một nhánh model chỉ được xem là hoàn thành khi:
- chạy được từ đầu đến cuối bằng script trong `scripts/`;
- không đọc nhầm test trong giai đoạn tuning;
- có đủ 2 feature representation;
- xuất file CSV kết quả;
- có phân tích ít nhất 5 False Positive và 5 False Negative;
- có nội dung giải thích phần mình phụ trách để đưa vào báo cáo;
- code không phá vỡ test chung trong `tests/`.
