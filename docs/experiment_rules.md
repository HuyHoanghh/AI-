# Experiment Contract

## Dataset
- `train.csv`: huấn luyện, tuning, cross-validation.
- `test.csv`: chỉ đánh giá cuối; không dùng tuning.
- `message`: input text.
- `label`: 0 = hợp lệ, 1 = spam/lừa đảo.

## Feature extraction
- CountVectorizer: `max_features=5000`
- TF-IDF: `max_features=5000`
- Fit vectorizer trên train, chỉ transform test.

## Models
- Multinomial Naive Bayes.
- KNN với k thuộc `[3,5,7,9,11,15,21]`.
- K của KNN được chọn bằng validation tách từ `train.csv`, không nhìn `test.csv`.

## Metrics
Accuracy, Precision, Recall, F1, Confusion Matrix, train time, prediction time.

## Fair comparison
Không tự thay MAX_FEATURES, preprocessing, label mapping hoặc test set trên feature branch.
