from common.metrics import evaluate_binary
from config.settings import KNN_K_VALUES
from models.knn.train_knn import train_knn


def tune_knn(X_train, y_train, X_test, y_test):
    rows = []
    for k in KNN_K_VALUES:
        model = train_knn(X_train, y_train, k=k)
        preds = model.predict(X_test)
        metrics = evaluate_binary(y_test, preds)
        rows.append({"k": k, **{m: metrics[m] for m in ["accuracy", "precision", "recall", "f1"]}})
    return rows
