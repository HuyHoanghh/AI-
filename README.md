# Vietnamese SMS Spam Detection

Đề tài phân loại tin nhắn tiếng Việt thành **Hợp lệ (0)** và **Spam/Lừa đảo (1)** bằng Machine Learning cơ bản.

## Dataset chính
Nguồn chính thức: https://github.com/trannguyenthaituan251209/vietnamese_sms_dataset

- `train.csv`: 2.394 mẫu, dùng huấn luyện/tuning/Cross-Validation.
- `test.csv`: 597 mẫu, **chỉ dùng đánh giá cuối**.
- Cột dùng chung: `message`, `label`.
- Không tự chia lại train/test trong thí nghiệm chính.

Tải tự động dataset chính thức:

```bash
python scripts/download_dataset.py
```

Hoặc tải thủ công và đặt tại:

```text
data/raw/train.csv
data/raw/test.csv
```

## Hai pipeline chạy song song

- `feature/naive-bayes`: Multinomial Naive Bayes
- `feature/knn`: K-Nearest Neighbors

Cả hai dùng chung `config/` và `common/` để đảm bảo cùng preprocessing, vectorizer và metrics.

Checklist chung: `docs/TEAM_CHECKLIST.md`.

## 4 cấu hình chính

1. Naive Bayes + CountVectorizer
2. Naive Bayes + TF-IDF
3. KNN + CountVectorizer
4. KNN + TF-IDF

Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix, thời gian train và predict.

## Tham khảo code
Repo tham khảo: https://github.com/Cham0703/DU_AN_SPAM

Các ý tưởng được tham khảo và viết lại cho project này gồm: xử lý giá trị rỗng, EDA độ dài tin nhắn, tần suất từ/WordCloud, quy tắc `fit` trên train và `transform` trên test, lưu model/vectorizer. Project **không sao chép pipeline tiếng Anh** của repo tham khảo: với SMS tiếng Việt, các token ẩn danh như `[PHONE]`, `[MONEY]`, `[DATE]`, `[TIME]` được giữ lại thay vì xóa số/ký tự một cách máy móc.

## Cấu trúc

```text
AI-/
├── config/
├── common/
├── data/raw/
├── models/naive_bayes/
├── models/knn/
├── scripts/
├── results/
├── saved_models/
├── app/
├── tests/
└── docs/
```

## Cài đặt

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_dataset.py
```

## Chạy

```bash
python scripts/run_nb.py
python scripts/run_knn.py
python scripts/compare_models.py
```

## Git workflow

```text
feature/naive-bayes ─┐
                     ├─> develop ─> main
feature/knn ─────────┘
```

Quy tắc:

- Không push trực tiếp vào `main`.
- Không sửa trực tiếp `common/` và `config/` trên feature branch nếu chưa thống nhất.
- Người 1 chỉ tập trung Naive Bayes; Người 2 chỉ tập trung KNN.
- Mỗi người hoàn thành branch -> tạo Pull Request vào `develop`.
- Chỉ merge `develop -> main` sau khi chạy integration test thành công.
