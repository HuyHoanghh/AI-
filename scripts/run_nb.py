from pathlib import Path
import joblib
from config.settings import TRAIN_PATH, TEST_PATH, TEXT_COLUMN, LABEL_COLUMN, PROJECT_ROOT
from common.data_loader import load_sms_csv, dataset_summary
from common.preprocessing import clean_series
from common.feature_extraction import build_count_vectorizer, build_tfidf_vectorizer, fit_transform_train_test
from common.metrics import classification_metrics, timed_fit_predict, result_row, save_results
from models.naive_bayes.train_nb import build_naive_bayes


def main():
    train = load_sms_csv(TRAIN_PATH)
    test = load_sms_csv(TEST_PATH)
    print("Train:", dataset_summary(train))
    print("Test :", dataset_summary(test))

    train_text = clean_series(train[TEXT_COLUMN])
    test_text = clean_series(test[TEXT_COLUMN])
    y_train, y_test = train[LABEL_COLUMN], test[LABEL_COLUMN]

    rows = []
    artifacts = PROJECT_ROOT / "saved_models"
    artifacts.mkdir(exist_ok=True)

    for feature_name, vectorizer in [
        ("Count", build_count_vectorizer()),
        ("TFIDF", build_tfidf_vectorizer()),
    ]:
        x_train, x_test = fit_transform_train_test(vectorizer, train_text, test_text)
        model = build_naive_bayes()
        pred, train_time, pred_time = timed_fit_predict(model, x_train, y_train, x_test)
        metrics = classification_metrics(y_test, pred)
        rows.append(result_row("NaiveBayes", feature_name, metrics, train_time, pred_time))
        print(feature_name, metrics)
        joblib.dump(model, artifacts / f"nb_{feature_name.lower()}.joblib")
        joblib.dump(vectorizer, artifacts / f"vectorizer_{feature_name.lower()}.joblib")

    out = PROJECT_ROOT / "results" / "naive_bayes" / "results_nb.csv"
    print(save_results(rows, out))


if __name__ == "__main__":
    main()
