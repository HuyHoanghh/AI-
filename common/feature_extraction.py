from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config.settings import MAX_FEATURES, NGRAM_RANGE


def build_count_vectorizer():
    return CountVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)


def build_tfidf_vectorizer():
    return TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)


def fit_transform_train_test(vectorizer, train_text, test_text):
    """Fit vocabulary only on train to prevent data leakage."""
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    return x_train, x_test
