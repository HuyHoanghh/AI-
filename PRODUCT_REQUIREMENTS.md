# YÊU CẦU SẢN PHẨM CUỐI — VIETNAMESE SMS SPAM DETECTION

Tài liệu này quy định các đầu ra bắt buộc của đồ án để hai thành viên phát triển song song nhưng khi merge vẫn đồng bộ.

## 1. Mục tiêu sản phẩm

Xây dựng hệ thống phân loại tin nhắn SMS tiếng Việt thành:

- `0` — Hợp lệ (Ham)
- `1` — Spam/Lừa đảo

Hai thuật toán bắt buộc phải được thực nghiệm và so sánh:

- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)

Hai cách biểu diễn văn bản bắt buộc:

- CountVectorizer
- TF-IDF

Tổng cộng phải có 4 cấu hình chính:

1. Naive Bayes + CountVectorizer
2. Naive Bayes + TF-IDF
3. KNN + CountVectorizer
4. KNN + TF-IDF

---

## 2. Yêu cầu dữ liệu

Nguồn dữ liệu chính thức:

`https://github.com/trannguyenthaituan251209/vietnamese_sms_dataset`

Dùng đúng hai file:

- `train.csv` — dùng huấn luyện, validation và tuning
- `test.csv` — chỉ dùng đánh giá cuối

Cột bắt buộc:

- `message`
- `label`

Không tự chia lại toàn bộ dataset trong thí nghiệm chính.

Không dùng `test.csv` để chọn hyperparameter.

---

## 3. Yêu cầu tiền xử lý

Pipeline chung phải:

- xử lý missing/NaN
- chuẩn hóa Unicode NFC
- chuyển chữ thường
- chuẩn hóa khoảng trắng
- chuẩn hóa URL thành token chung
- giữ lại các token ẩn danh có ý nghĩa như `[PHONE]`, `[BANK_ACC]`, `[MONEY]`, `[NUMBER]`, `[TIME]`, `[DATE]`
- không xóa toàn bộ chữ số một cách máy móc
- không dùng English stopwords cho dữ liệu tiếng Việt

Hai thành viên phải dùng cùng một hàm preprocessing trong `common/preprocessing.py`.

---

## 4. Yêu cầu EDA

Sản phẩm EDA tối thiểu phải có:

- số lượng mẫu train/test
- phân bố Ham/Spam
- số missing values
- số duplicate
- thống kê độ dài tin nhắn
- số từ/tin nhắn
- top từ phổ biến của Ham
- top từ phổ biến của Spam
- phân tích tần suất các token `[PHONE]`, `[MONEY]`, `[DATE]`, `[TIME]`, ...
- ít nhất 2 biểu đồ phục vụ báo cáo

Có thể tham khảo cách làm EDA từ repo:

`https://github.com/Cham0703/DU_AN_SPAM`

nhưng không sao chép preprocessing tiếng Anh của repo đó.

---

## 5. Yêu cầu Người 1 — Naive Bayes

Branch làm việc:

`feature/naive-bayes`

Phải hoàn thành:

- chạy CountVectorizer
- chạy TF-IDF
- train `MultinomialNB`
- đánh giá 2 cấu hình
- tính Accuracy, Precision, Recall, F1-score
- tạo Confusion Matrix cho từng cấu hình
- đo training time
- đo prediction time
- thực hiện error analysis
- lấy ví dụ False Positive
- lấy ví dụ False Negative
- có thể chạy 5-Fold Cross-Validation trên `train.csv`

### File đầu ra bắt buộc

```text
models/naive_bayes/
├── train_nb.py
└── evaluate_nb.py

notebooks/member1/
└── naive_bayes_experiment.ipynb

results/naive_bayes/
├── results_nb.csv
├── confusion_matrix_nb_count.png
└── confusion_matrix_nb_tfidf.png
```

`results_nb.csv` phải có tối thiểu:

```text
model,feature,accuracy,precision,recall,f1,training_time,prediction_time
```

---

## 6. Yêu cầu Người 2 — KNN

Branch làm việc:

`feature/knn`

Phải hoàn thành:

- chạy CountVectorizer
- chạy TF-IDF
- train KNN
- tune K
- thử tối thiểu các giá trị `k = 3, 5, 7, 9, 11, 15, 21`
- chọn K tốt nhất trên dữ liệu train/validation
- tuyệt đối không dùng test để chọn K
- đánh giá cấu hình tốt nhất của Count và TF-IDF
- tính Accuracy, Precision, Recall, F1-score
- tạo Confusion Matrix
- đo training time
- đo prediction time
- vẽ biểu đồ K so với F1-score
- thực hiện error analysis

### File đầu ra bắt buộc

```text
models/knn/
├── train_knn.py
├── tune_knn.py
└── evaluate_knn.py

notebooks/member2/
└── knn_experiment.ipynb

results/knn/
├── results_knn.csv
├── knn_k_comparison.csv
├── knn_k_chart.png
├── confusion_matrix_knn_count.png
└── confusion_matrix_knn_tfidf.png
```

`results_knn.csv` phải có tối thiểu:

```text
model,feature,accuracy,precision,recall,f1,training_time,prediction_time,k
```

---

## 7. Yêu cầu kết quả cuối cùng

Sau khi merge hai branch vào `develop`, phải có:

```text
results/final/final_comparison.csv
```

Bảng cuối phải so sánh tối thiểu:

| Model | Feature | Accuracy | Precision | Recall | F1 | Train Time | Predict Time |
|---|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | Count | ... | ... | ... | ... | ... | ... |
| Naive Bayes | TF-IDF | ... | ... | ... | ... | ... | ... |
| KNN | Count | ... | ... | ... | ... | ... | ... |
| KNN | TF-IDF | ... | ... | ... | ... | ... | ... |

Báo cáo phải trả lời được:

1. Naive Bayes hay KNN tốt hơn trên bộ dữ liệu này?
2. CountVectorizer hay TF-IDF tốt hơn?
3. Mô hình nào có Recall tốt hơn cho lớp Spam?
4. Mô hình nào có F1-score tốt nhất?
5. Mô hình nào train/predict nhanh hơn?
6. Các lỗi False Positive và False Negative thường thuộc kiểu tin nhắn nào?

---

## 8. Yêu cầu model và artifact

Sau khi chọn được mô hình tốt nhất, phải lưu:

```text
saved_models/
├── best_model.joblib
└── best_vectorizer.joblib
```

Ứng dụng demo phải dùng đúng preprocessing + vectorizer + model đã train.

---

## 9. Yêu cầu demo

Demo tối thiểu phải cho phép:

- nhập một tin nhắn tiếng Việt
- bấm nút phân loại
- trả kết quả `Ham` hoặc `Spam/Lừa đảo`
- hiển thị rõ kết quả

Khuyến nghị dùng Streamlit:

```text
app/app.py
```

Luồng demo:

```text
Tin nhắn người dùng
→ preprocessing
→ vectorizer
→ best model
→ Ham / Spam
```

---

## 10. Yêu cầu báo cáo

Báo cáo tối thiểu nên có:

1. Giới thiệu bài toán Spam Detection
2. Tổng quan NLP và Text Classification
3. Giới thiệu Vietnamese SMS Dataset
4. Phân tích dữ liệu (EDA)
5. Tiền xử lý dữ liệu
6. CountVectorizer và TF-IDF
7. Multinomial Naive Bayes
8. KNN và K tuning
9. Thiết kế thực nghiệm
10. Kết quả Naive Bayes
11. Kết quả KNN
12. So sánh 4 cấu hình
13. Error Analysis
14. Demo sản phẩm
15. Kết luận và hạn chế

---

## 11. Yêu cầu Git/GitHub

Không push trực tiếp code cá nhân lên `main`.

Luồng chuẩn:

```text
feature/naive-bayes ─┐
                     ├──> develop ───> main
feature/knn ─────────┘
```

Mỗi thành viên phải:

- commit có nội dung rõ ràng
- push lên đúng feature branch
- không tự ý sửa `common/` hoặc `config/` nếu chưa thống nhất
- trước khi tạo Pull Request phải chạy test
- Pull Request phải target vào `develop`

Ví dụ commit:

```text
feat(nb): complete naive bayes evaluation
feat(knn): add k tuning experiment
fix(preprocessing): preserve anonymized tokens
```

---

## 12. Điều kiện được xem là HOÀN THÀNH

Project chỉ được xem là hoàn thành khi:

- [ ] Dataset tải và đọc được
- [ ] Test preprocessing pass
- [ ] Naive Bayes chạy được với Count
- [ ] Naive Bayes chạy được với TF-IDF
- [ ] KNN chạy được với Count
- [ ] KNN chạy được với TF-IDF
- [ ] Có tuning K
- [ ] Có đủ Accuracy / Precision / Recall / F1
- [ ] Có Confusion Matrix
- [ ] Có Error Analysis
- [ ] Có `results_nb.csv`
- [ ] Có `results_knn.csv`
- [ ] Có `final_comparison.csv`
- [ ] Chọn được best model
- [ ] Lưu model + vectorizer
- [ ] Demo nhập tin nhắn và dự đoán được
- [ ] Hai feature branch merge sạch vào `develop`
- [ ] `develop` chạy ổn trước khi merge vào `main`

---

## 13. Sản phẩm bàn giao cuối cùng

Bộ sản phẩm cuối phải gồm:

```text
1. Source code trên GitHub
2. Dataset source/link
3. Notebook thực nghiệm Naive Bayes
4. Notebook thực nghiệm KNN
5. Kết quả CSV của từng model
6. Confusion Matrix
7. Biểu đồ tuning K
8. Final comparison table
9. Best trained model
10. Best vectorizer
11. Demo ứng dụng
12. Báo cáo Word/PDF
13. Slide thuyết trình
14. README hướng dẫn cài và chạy
```

Đây là checklist nghiệm thu chính thức của project.