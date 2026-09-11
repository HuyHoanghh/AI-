"""Entry point cho thí nghiệm KNN (Người 2).

Luồng thí nghiệm đúng quy tắc chung (docs/experiment_rules.md):
1. Load `train.csv` và `test.csv` chính thức.
2. Tuning K bằng 5-fold CV chỉ trên `train.csv` (KHÔNG nhìn `test.csv`).
3. Chọn best K cho riêng Count và TF-IDF.
4. Train lại KNN tốt nhất trên toàn bộ train, đánh giá một lần trên test.
5. Xuất toàn bộ artifact vào `results/knn/`.

Chạy từ project root:
    python scripts/run_knn.py
"""
import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python scripts/run_knn.py`: thêm project root
# vào sys.path để import được các package nội bộ (config, common, models).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Console Windows mặc định dùng cp1252 — ép UTF-8 để in tiếng Việt không lỗi.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from config.settings import (
    LABEL_COLUMN,
    PROJECT_ROOT,
    TEST_PATH,
    TEXT_COLUMN,
    TRAIN_PATH,
)
from common.data_loader import dataset_summary, load_sms_csv
from common.metrics import save_results
from common.preprocessing import clean_series
from models.knn.evaluate_knn import run_knn_experiment
from models.knn.tune_knn import pick_best_k, tune_k_cross_val


def main():
    train = load_sms_csv(TRAIN_PATH)
    test = load_sms_csv(TEST_PATH)
    print("TRAIN:", dataset_summary(train))
    print("TEST :", dataset_summary(test))

    train_text = clean_series(train[TEXT_COLUMN])
    test_text = clean_series(test[TEXT_COLUMN])
    y_train = train[LABEL_COLUMN]
    y_test = test[LABEL_COLUMN]

    # Kiểm tra nhanh 10 mẫu trước/sau preprocessing (checklist mục C).
    print("\n=== 10 mẫu trước/sau preprocessing ===")
    for raw, cleaned in list(zip(train[TEXT_COLUMN], train_text))[:10]:
        raw_s = " ".join(str(raw).split())[:90]
        print(f"- RAW: {raw_s}")
        print(f"  CLN: {cleaned[:90]}")

    # --- Step 1: Tuning K only on train (5-fold CV, never test.csv) ------
    print("\n=== Tuning K with 5-fold CV on train.csv ===")
    tune_rows = tune_k_cross_val(train_text, y_train)

    best_k_by_feature = {}
    for feature_name in ("Count", "TFIDF"):
        k, f1 = pick_best_k(tune_rows, feature_name)
        best_k_by_feature[feature_name] = k
        print(f"Best K for {feature_name}: k={k} (CV F1 = {f1:.4f})")

    # --- Step 2: Refit on full train, evaluate once on test.csv ----------
    print("\n=== Evaluation on test.csv (single final use) ===")
    rows, _ = run_knn_experiment(
        test_df=test,
        train_text=train_text,
        train_labels=y_train,
        test_text=test_text,
        test_labels=y_test,
        best_k_by_feature=best_k_by_feature,
        tune_rows=tune_rows,
    )

    out = PROJECT_ROOT / "results" / "knn" / "results_knn.csv"
    print("\n=== Bảng kết quả KNN ===")
    print(save_results(rows, out).to_string(index=False))
    print("\nHoàn tất thí nghiệm KNN.")


if __name__ == "__main__":
    main()
