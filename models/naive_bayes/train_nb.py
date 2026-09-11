from sklearn.naive_bayes import MultinomialNB


def build_naive_bayes(alpha: float = 1.0):
    return MultinomialNB(alpha=alpha)
