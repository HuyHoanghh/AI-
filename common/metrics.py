from time import perf_counter
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def classification_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def timed_fit_predict(model, x_train, y_train, x_test):
    start = perf_counter()
    model.fit(x_train, y_train)
    training_time = perf_counter() - start

    start = perf_counter()
    prediction = model.predict(x_test)
    prediction_time = perf_counter() - start
    return prediction, training_time, prediction_time


def result_row(model_name, feature_name, metrics, training_time, prediction_time):
    return {
        "model": model_name,
        "feature": feature_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "training_time": training_time,
        "prediction_time": prediction_time,
    }


def save_results(rows, path):
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
