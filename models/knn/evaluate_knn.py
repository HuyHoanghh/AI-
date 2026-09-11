from common.metrics import evaluate_binary


def evaluate_knn(model, X_test, y_test):
    preds = model.predict(X_test)
    return evaluate_binary(y_test, preds), preds
