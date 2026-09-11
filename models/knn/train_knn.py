from sklearn.neighbors import KNeighborsClassifier


def build_knn(k: int):
    return KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
