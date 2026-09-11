from common.metrics import classification_metrics


def evaluate_nb(model, X_test, y_test):
    preds = model.predict(X_test)
    return classification_metrics(y_test, preds), preds
