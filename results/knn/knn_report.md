# Báo cáo phần KNN (Người 2)

## 1. Mục tiêu

Phân loại tin nhắn SMS tiếng Việt thành **Ham (0)** / **Spam (1)** bằng mô hình
**K-Nearest Neighbors**, với 2 cách biểu diễn đặc trưng:

1. KNN + **CountVectorizer**
2. KNN + **TF-IDF Vectorizer**

Toàn bộ preprocessing và vectorization dùng chung code trong `common/` và
`config/settings.py` để so sánh công bằng với mô hình Naive Bayes của Người 1.

## 2. Dữ liệu

- `data/raw/train.csv`: 2394 mẫu (dùng để tuning + huấn luyện cuối).
- `data/raw/test.csv`: 597 mẫu (chỉ dùng **một lần duy nhất** để đánh giá).
- Cột: `message` (nội dung), `label` (0 = Ham, 1 = Spam).
- Không có missing; **có duplicate** theo cặp `(message, label)`:
  **303 dòng trùng ở train** và **31 dòng trùng ở test**
  (đo bằng `common.data_loader.dataset_summary`).
- Cách xử lý: **giữ nguyên toàn bộ dữ liệu gốc** — không xóa duplicate,
  không chia lại train/test. Dataset chính thức là tài sản chung của nhóm
  (mọi thành viên dùng chung một bản), việc thay đổi phải thống nhất với
  cả nhóm theo `docs/TEAM_CHECKLIST.md`. Lưu ý: duplicate làm các mẫu lặp
  xuất hiện ở cả train và test nên KNN có xu hướng "nhớ" mẫu đã gặp —
  đây là đặc điểm của dataset, ảnh hưởng như nhau đến mọi model khi so sánh.

## 3. Preprocessing (dùng chung `common.preprocessing.clean_series`)

- Chuẩn hóa Unicode NFC, đưa về chữ thường.
- Thay URL/email bằng token đặc biệt `__url__`, `__email__`.
- Giữ lại các token đã ẩn danh hóa trong dataset
  (`[PHONE]` → `__phone__`, `[MONEY]` → `__money__`, ...) vì chúng là tín hiệu
  quan trọng của tin lừa đảo/quảng cáo.
- Giữ lại chữ số, chỉ loại ký tự đặc biệt vô nghĩa.

## 4. Vectorization

| Tham số | Giá trị (theo `config/settings.py`) |
|---|---|
| `max_features` | 5000 |
| `ngram_range` | (1, 1) — unigram |
| Nguyên tắc | `fit` trên train, chỉ `transform` test (chống data leakage) |

Shape ma trận thu được: `X_train = (2394, 4346)`, `X_test = (597, 4346)`.

## 5. Chọn khoảng cách và tuning K

- KNN dùng **khoảng cách cosine** (`metric="cosine"`, `algorithm="brute"`),
  phù hợp với vector text thưa, chiều cao — cosine đo hướng của vector thay vì
  độ lớn, tránh việc tin nhắn dài bị coi là "xa" mọi tin khác.
- K được chọn bằng **Stratified 5-Fold Cross-Validation chỉ trên `train.csv`**
  với K ∈ {3, 5, 7, 9, 11, 15, 21}. Tiêu chí chọn: **F1 trung bình** cao nhất
  (lớp Spam là lớp dương), hòa thì chọn K nhỏ hơn.
- **Chống rò rỉ dữ liệu trong CV:** vectorizer + KNN được gói trong
  `sklearn.pipeline.Pipeline` và cross-validation chạy **trên văn bản đã
  preprocessing** — mỗi fold tự fit vectorizer (từ vựng / trọng số IDF) chỉ
  trên phần train của fold đó, không fold nào "nhìn thấy" phần validation
  của mình trong bước vectorization.
- `test.csv` **không hề được nhìn đến** trong bước tuning.

### Bảng tuning K (CV F1 trên train.csv, pipeline chống rò rỉ)

| K | Count F1 (±std) | Count Acc | TF-IDF F1 (±std) | TF-IDF Acc |
|---|---|---|---|---|
| **3**  | **0.9128 ± 0.0226** | 0.9557 | **0.9000 ± 0.0127** | 0.9495 |
| 5  | 0.9031 ± 0.0199 | 0.9511 | 0.8815 ± 0.0129 | 0.9415 |
| 7  | 0.8896 ± 0.0238 | 0.9449 | 0.8496 ± 0.0293 | 0.9269 |
| 9  | 0.8757 ± 0.0307 | 0.9390 | 0.8351 ± 0.0301 | 0.9211 |
| 11 | 0.8650 ± 0.0333 | 0.9344 | 0.8300 ± 0.0257 | 0.9190 |
| 15 | 0.8343 ± 0.0333 | 0.9215 | 0.8072 ± 0.0376 | 0.9102 |
| 21 | 0.8154 ± 0.0453 | 0.9148 | 0.7768 ± 0.0339 | 0.8989 |

Lưu ý: số của Count không đổi so với cách làm cũ vì bộ từ vựng đầy đủ chỉ có
4346 token (< `max_features=5000`) nên từ vựng mỗi fold gần như trùng nhau;
TF-IDF thay đổi nhẹ vì trọng số IDF phụ thuộc tần suất tài liệu của từng fold.

**Kết luận tuning:** Best K = **3** cho cả Count và TF-IDF. F1 giảm đơn điệu khi
K tăng — với tập train tương đối nhỏ (2394 mẫu), K lớn làm "pha loãng" tín hiệu
của lớp Spam vào các láng giềng Ham đông hơn. Biểu đồ: `results/knn/knn_k_chart.png`.

## 6. Kết quả đánh giá trên test.csv (597 mẫu, dùng 1 lần duy nhất)

| Model | Feature | Accuracy | Precision | Recall | F1 | Train time | Predict time | K |
|---|---|---|---|---|---|---|---|---|
| KNN | Count | 0.9397 | 0.8631 | 0.9177 | **0.8896** | ~2 ms | ~150 ms | 3 |
| KNN | TF-IDF | 0.9296 | 0.8537 | 0.8861 | **0.8696** | ~3 ms | ~150 ms | 3 |

Confusion matrix (Count, k=3): TP = 145, FP = 23, TN = 416, FN = 13.
Confusion matrix (TF-IDF, k=3): TP = 140, FP = 24, TN = 415, FN = 18.

Ảnh: `results/knn/confusion_matrix_knn_count.png`,
`results/knn/confusion_matrix_knn_tfidf.png`.

### So sánh Count vs TF-IDF

- **Count nhỉnh hơn TF-IDF** ở cả 4 chỉ số (F1 0.8896 vs 0.8696).
- Lý do: spam SMS tiếng Việt lặp lại nhiều từ khóa đặc trưng
  (ví dụ `vay`, `nap`, `nhan`, `dang ky`, tên nhà cái...). Vector đếm giữ được
  tần suất lặp này, còn TF-IDF hạ trọng số các từ xuất hiện phổ biến ở cả 2 lớp
  nên làm mờ bớt tín hiệu.
- Chênh lệch không lớn (~2 điểm F1), cả 2 cấu hình đều dùng tốt.

### Runtime

- Training gần như tức thời (vài ms) vì KNN là lazy learner — chỉ lưu dữ liệu.
- Prediction ~150 ms cho 597 mẫu (brute-force cosine, `n_jobs=-1`; số đo dao động
  theo máy). Đây là nhược điểm thực tế của KNN: chi phí dự đoán tăng tuyến tính
  theo kích thước tập train, ngược với Naive Bayes dự đoán gần như tức thời.

## 7. Error analysis (chi tiết: `results/knn/error_analysis_knn.md`)

Mỗi cấu hình trích 5 FP + 5 FN từ test set:

- **False Positive** (Ham → đoán Spam): chủ yếu là tin Ham mang phong cách
  thông báo/quảng bá (điện lực, mã giảm giá Lotte Cinema, Ahamove, bảo hành...)
  chứa `[MONEY]`, link, lời mời — nằm "gần" cụm Spam trong không gian vector.
- **False Negative** (Spam → đoán Ham): chủ yếu là spam dùng chữ biến dạng cố
  tình né bộ lọc (`G@me bài tra thu0ng kie'mtien`, `tr0`, `t0c`), spam tín
  dụng/legal đe dọa dài dòng giống văn phong hành chính, hoặc spam cờ bạc ngắn
  (`nk99hv.com`) — các token biến dạng không khớp từ vựng spam điển hình nên
  đa số láng giềng bỏ phiếu Ham.
- Hướng cải thiện: thêm n-gram (2,3) để bắt cụm từ, chuẩn hóa chữ biến dạng
  (telex/leet), hoặc thử metric/weighting khác (`weights="distance"`).

## 8. Artifacts

| File | Nội dung |
|---|---|
| `scripts/run_knn.py` | Script chạy toàn bộ thí nghiệm từ đầu đến cuối |
| `models/knn/train_knn.py` | Build KNN (cosine, brute) |
| `models/knn/tune_knn.py` | Tuning K bằng Stratified 5-Fold CV trên train |
| `models/knn/evaluate_knn.py` | Đánh giá trên test, vẽ biểu đồ, error analysis |
| `notebooks/member2/knn_experiment.ipynb` | Notebook giải thích từng bước thí nghiệm |
| `results/knn/results_knn.csv` | Bảng kết quả 2 cấu hình (đúng schema chung) |
| `results/knn/knn_k_comparison.csv` | Bảng tuning K → F1 |
| `results/knn/knn_k_chart.png` | Biểu đồ K vs F1 (mean ± std) |
| `results/knn/confusion_matrix_knn_count.png` | Confusion matrix Count |
| `results/knn/confusion_matrix_knn_tfidf.png` | Confusion matrix TF-IDF |
| `results/knn/error_analysis_knn.md` | 5 FP + 5 FN mỗi cấu hình + nhận xét |

## 9. Kết luận

- **KNN + CountVectorizer (k=3) là cấu hình tốt nhất**: F1 = 0.8896,
  Recall = 0.9177 (bắt được ~92% tin Spam), Accuracy = 0.9397.
- Quy trình tuân thủ đúng quy tắc thực nghiệm: chọn K chỉ bằng CV trên
  `train.csv`, vectorizer fit trên train, `test.csv` chỉ dùng một lần cuối.
- Đổi lại recall cao, KNN tốn chi phí dự đoán hơn hẳn mô hình generative
  (Naive Bayes) — sẽ được so sánh tổng thể trong `results/final/final_comparison.csv`.
