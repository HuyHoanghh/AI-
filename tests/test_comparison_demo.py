import joblib
import pandas as pd
import pytest
from app.inference import compare_message
from common.preprocessing import clean_series
from common.metrics import classification_metrics
from config.settings import PROJECT_ROOT, TEST_PATH
from scripts.export_demo_model import export_model


def test_comparison_metrics_and_neighbor_votes(tmp_path):
    bundle = joblib.load(export_model(tmp_path) / 'demo_comparison.joblib')
    test = pd.read_csv(TEST_PATH)
    report = pd.read_csv(PROJECT_ROOT / 'results/final/final_comparison.csv')
    for name, (model, vectorizer) in bundle['configurations'].items():
        algorithm, feature = name.split(' + ')
        predicted = model.predict(vectorizer.transform(clean_series(test.message)))
        metrics = classification_metrics(test.label, predicted)
        expected = report[(report.model == algorithm) & (report.feature == feature)].iloc[0]
        assert metrics['f1'] == pytest.approx(expected.f1)
    rows, neighbors = compare_message(test.message.iloc[-1], bundle)
    assert len(rows) == 4
    for name, items in neighbors.items():
        assert len(items) == 3
        assert [i['distance'] for i in items] == sorted(i['distance'] for i in items)
        vote = 'Spam / Lừa đảo' if sum(i['label'] for i in items) >= 2 else 'Hợp lệ (Ham)'
        assert next(r['Dự đoán'] for r in rows if r['Cấu hình'] == name) == vote
