"""Tune K cho KNN trên `train.csv` bằng Stratified K-Fold cross-validation.

Quy tắc thực nghiệm (docs/experiment_rules.md):
- K được chọn bằng validation tách từ `train.csv`, KHÔNG nhìn `test.csv`.
- Thử k thuộc KNN_K_VALUES = [3, 5, 7, 9, 11, 15, 21].
- Dùng chung config vectorizer (MAX_FEATURES, NGRAM_RANGE) cho cả 2 thành viên.

Kết quả trả về bảng [{feature, k, f1_mean, f1_std, accuracy_mean}] để script
`scripts/run_knn.py` xuất ra `results/knn/knn_k_comparison.csv`.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config.settings import CV_FOLDS, KNN_K_VALUES, RANDOM_STATE
from common.feature_extraction import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
)
from models.knn.train_knn import build_knn


def tune_k_cross_val(train_text, y_train, k_values=None, n_splits=CV_FOLDS):
    """Chạy tuning K cho cả Count và TF-IDF, trả về bảng kết quả.

    Mỗi (feature, k) được đánh giá bằng Stratified K-Fold trên toàn bộ
    `train.csv`; điểm chọn model là F1 trung bình (lớp Spam là lớp dương).
    """
    k_values = k_values or KNN_K_VALUES
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for feature_name, builder in [
        ("Count", build_count_vectorizer),
        ("TFIDF", build_tfidf_vectorizer),
    ]:
        # Fit vocabulary trên TOÀN BỘ train.csv (không đụng test.csv).
        vectorizer = builder()
        X = vectorizer.fit_transform(train_text)

        for k in k_values:
            model = build_knn(k)
            f1_scores = cross_val_score(
                model, X, y_train, cv=cv, scoring="f1", n_jobs=1
            )
            acc_scores = cross_val_score(
                model, X, y_train, cv=cv, scoring="accuracy", n_jobs=1
            )
            rows.append(
                {
                    "feature": feature_name,
                    "k": k,
                    "f1_mean": float(np.mean(f1_scores)),
                    "f1_std": float(np.std(f1_scores)),
                    "accuracy_mean": float(np.mean(acc_scores)),
                }
            )
            print(
                f"[{feature_name}] k={k:>2}  "
                f"f1={rows[-1]['f1_mean']:.4f} ± {rows[-1]['f1_std']:.4f}  "
                f"acc={rows[-1]['accuracy_mean']:.4f}"
            )
    return rows


def pick_best_k(rows, feature_name):
    """Chọn K có F1 trung bình cao nhất cho một feature representation."""
    subset = [r for r in rows if r["feature"] == feature_name]
    best = max(subset, key=lambda r: (r["f1_mean"], -r["k"]))
    return best["k"], best["f1_mean"]