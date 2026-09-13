from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from app.inference import load_artifacts, predict_message
from scripts.export_demo_model import export_model
from common.preprocessing import clean_series
from common.metrics import classification_metrics
from config.settings import PROJECT_ROOT, TEST_PATH


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    return load_artifacts(export_model(tmp_path_factory.mktemp('demo')))


def test_saved_model_matches_report(artifacts):
    model, vectorizer = artifacts
    test = pd.read_csv(TEST_PATH)
    prediction = model.predict(vectorizer.transform(clean_series(test.message)))
    metrics = classification_metrics(test.label, prediction)
    row = pd.read_csv(PROJECT_ROOT / 'results/final/final_comparison.csv').query(
        "model == 'KNN' and feature == 'Count'").iloc[0]
    for metric in ('accuracy', 'precision', 'recall', 'f1'):
        assert metrics[metric] == pytest.approx(row[metric])
    assert np.array_equal(metrics['confusion_matrix'], [[416, 23], [13, 145]])


@pytest.mark.parametrize('text', ['', '   ', '!!!', 'zzzzqwertyxxxx'])
def test_rejects_uninformative_input(artifacts, text):
    with pytest.raises(ValueError):
        predict_message(text, *artifacts)


def test_ui_predict_and_empty_warning(artifacts, monkeypatch, tmp_path):
    import app.inference as inference
    import joblib
    joblib.dump(artifacts[0], tmp_path / 'best_model.joblib')
    joblib.dump(artifacts[1], tmp_path / 'best_vectorizer.joblib')
    monkeypatch.setattr(inference, 'ARTIFACT_DIR', tmp_path)
    monkeypatch.setattr(inference, 'load_artifacts', lambda: artifacts)
    at = AppTest.from_file(str(PROJECT_ROOT / 'app/app.py'), default_timeout=20).run()
    assert not at.exception
    at.button[-1].click().run()
    assert at.warning
    test = pd.read_csv(TEST_PATH)
    for label in (0, 1):
        message = test.loc[test.label == label, 'message'].iloc[-1]
        expected, _ = predict_message(message, *artifacts)
        at.text_area[0].set_value(message)
        at.button[-1].click().run()
        assert not at.exception
        output = at.error if expected else at.success
        assert output and 'Dự đoán:' in output[0].value
