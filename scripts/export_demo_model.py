"""Export the frozen KNN Count k=3 configuration selected with train-only CV."""
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import sklearn
from common.data_loader import load_sms_csv
from common.feature_extraction import build_count_vectorizer, build_tfidf_vectorizer
from models.naive_bayes.train_nb import build_naive_bayes
from common.preprocessing import clean_series
from config.settings import PROJECT_ROOT, TRAIN_PATH, TEXT_COLUMN, LABEL_COLUMN
from models.knn.train_knn import build_knn


def export_model(output_dir=None):
    train = load_sms_csv(TRAIN_PATH)
    vectorizer = build_count_vectorizer()
    x = vectorizer.fit_transform(clean_series(train[TEXT_COLUMN]))
    model = build_knn(3)
    model.fit(x, train[LABEL_COLUMN])
    out = Path(output_dir or PROJECT_ROOT / 'saved_models')
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / 'best_model.joblib')
    joblib.dump(vectorizer, out / 'best_vectorizer.joblib')
    configurations = {}
    for feature, builder in [('Count', build_count_vectorizer), ('TFIDF', build_tfidf_vectorizer)]:
        v = builder()
        matrix = v.fit_transform(clean_series(train[TEXT_COLUMN]))
        for name, estimator in [('NaiveBayes', build_naive_bayes()), ('KNN', build_knn(3))]:
            estimator.fit(matrix, train[LABEL_COLUMN])
            configurations[f'{name} + {feature}'] = (estimator, v)
    # Keep training messages in the exact fitted row order for neighbor explanations.
    joblib.dump({'configurations': configurations,
                 'messages': train[TEXT_COLUMN].tolist(),
                 'labels': train[LABEL_COLUMN].tolist()}, out / 'demo_comparison.joblib')
    metadata = {'model': 'KNN', 'feature': 'Count', 'k': 3,
                'selection': 'Frozen configuration from train-only 5-fold CV',
                'train_rows': len(train), 'features': x.shape[1],
                'train_sha256': hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
                'sklearn': sklearn.__version__}
    (out / 'demo_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    print(export_model())
