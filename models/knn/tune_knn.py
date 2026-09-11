from sklearn.metrics import f1_score
from models.knn.train_knn import build_knn


def tune_k(x_train, y_train, x_valid, y_valid, k_values):
    scores = []
    for k in k_values:
        model = build_knn(k)
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        scores.append({"k": k, "f1": f1_score(y_valid, pred, zero_division=0)})
    return scores
