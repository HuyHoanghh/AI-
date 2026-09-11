from sklearn.neighbors import KNeighborsClassifier


def build_knn(k: int, metric: str = "cosine") -> KNeighborsClassifier:
    """Build a KNN classifier.

    `cosine` distance is used because it works well with high-dimensional
    sparse text vectors; `brute` is exact and fast enough for this dataset.
    """
    return KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )