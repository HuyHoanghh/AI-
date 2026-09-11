from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config.settings import MAX_FEATURES


def build_count_vectorizer():
    return CountVectorizer(max_features=MAX_FEATURES)


def build_tfidf_vectorizer():
    return TfidfVectorizer(max_features=MAX_FEATURES)
