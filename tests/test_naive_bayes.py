import numpy as np
import pandas as pd
from common.feature_extraction import build_count_vectorizer
from models.naive_bayes.train_nb import build_naive_bayes
from models.naive_bayes.evaluate_nb import evaluate_nb
from models.naive_bayes.experiment import error_table, cross_validate_feature


def test_evaluate_nb_and_unseen_test_word():
    vectorizer = build_count_vectorizer()
    x = vectorizer.fit_transform(["hello friend", "win prize", "hello meeting", "win cash"])
    model = build_naive_bayes().fit(x, [0, 1, 0, 1])
    metrics, pred = evaluate_nb(model, vectorizer.transform(["hello unseenword", "win"]), [0, 1])
    assert "unseenword" not in vectorizer.vocabulary_
    assert list(pred) == [0, 1]
    assert metrics["f1"] == 1


def test_error_table_preserves_source_rows():
    df = pd.DataFrame({"message": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]},
                      index=[10, 20, 30, 40])
    errors = error_table(df, ["a", "b", "c", "d"], np.array([1, 0, 0, 1]))
    assert errors.test_row.tolist() == [0, 1]
    assert errors.error_type.tolist() == ["FP", "FN"]
    assert error_table(df, df.message, df.label.to_numpy()).empty


def test_cv_fits_vocabulary_inside_each_fold(monkeypatch):
    from sklearn.feature_extraction.text import CountVectorizer
    seen = []
    original = CountVectorizer.fit_transform

    def recording_fit(self, texts, y=None):
        texts = list(texts)
        seen.append(len(texts))
        return original(self, texts, y)

    monkeypatch.setattr(CountVectorizer, "fit_transform", recording_fit)
    text = pd.Series(["hello friend", "win prize"] * 10)
    scores = cross_validate_feature(text, pd.Series([0, 1] * 10), "Count")
    assert len(scores) == 5
    assert seen == [16] * 5
