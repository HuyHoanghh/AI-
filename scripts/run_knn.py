from sklearn.model_selection import train_test_split
from config.settings import TRAIN_PATH, TEST_PATH, TEXT_COLUMN, LABEL_COLUMN, PROJECT_ROOT, KNN_K_VALUES, RANDOM_STATE
from common.data_loader import load_sms_csv, dataset_summary
from common.preprocessing import clean_series
from common.feature_extraction import build_count_vectorizer, build_tfidf_vectorizer, fit_transform_train_test
from common.metrics import classification_metrics, timed_fit_predict, result_row, save_results
from models.knn.train_knn import build_knn


def choose_k(train_text, y_train, vectorizer):
    text_a, text_b, y_a, y_b = train_test_split(
        train_text, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    x_a = vectorizer.fit_transform(text_a)
    x_b = vectorizer.transform(text_b)
    best = None
    for k in KNN_K_VALUES:
        model = build_knn(k)
        model.fit(x_a, y_a)
        score = classification_metrics(y_b, model.predict(x_b))["f1"]
        print(f"k={k}, validation_f1={score:.4f}")
        if best is None or score > best[1]:
            best = (k, score)
    return best[0]


def main():
    train = load_sms_csv(TRAIN_PATH)
    test = load_sms_csv(TEST_PATH)
    print("Train:", dataset_summary(train))
    print("Test :", dataset_summary(test))

    train_text = clean_series(train[TEXT_COLUMN])
    test_text = clean_series(test[TEXT_COLUMN])
    y_train, y_test = train[LABEL_COLUMN], test[LABEL_COLUMN]
    rows = []

    for feature_name, builder in [
        ("Count", build_count_vectorizer),
        ("TFIDF", build_tfidf_vectorizer),
    ]:
        k = choose_k(train_text, y_train, builder())
        vectorizer = builder()
        x_train, x_test = fit_transform_train_test(vectorizer, train_text, test_text)
        model = build_knn(k)
        pred, train_time, pred_time = timed_fit_predict(model, x_train, y_train, x_test)
        metrics = classification_metrics(y_test, pred)
        row = result_row("KNN", feature_name, metrics, train_time, pred_time)
        row["k"] = k
        rows.append(row)
        print(feature_name, "best_k=", k, metrics)

    out = PROJECT_ROOT / "results" / "knn" / "results_knn.csv"
    print(save_results(rows, out))


if __name__ == "__main__":
    main()
