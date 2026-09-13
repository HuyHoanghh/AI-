"""Shared inference logic for the demo and its tests."""
from pathlib import Path
import joblib
from common.preprocessing import clean_message

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / 'saved_models'


def load_artifacts(directory=ARTIFACT_DIR):
    directory = Path(directory)
    model = joblib.load(directory / 'best_model.joblib')
    vectorizer = joblib.load(directory / 'best_vectorizer.joblib')
    if model.n_features_in_ != len(vectorizer.vocabulary_):
        raise ValueError('Model và vectorizer không khớp. Hãy xuất lại cùng một lần.')
    return model, vectorizer


def predict_message(message, model, vectorizer):
    cleaned = clean_message(message)
    if not cleaned:
        raise ValueError('Vui lòng nhập tin nhắn có chữ hoặc số để phân loại.')
    x = vectorizer.transform([cleaned])
    if x.nnz == 0:
        raise ValueError('Tin nhắn chưa có từ nào trong từ vựng của mô hình. Hãy nhập nội dung đầy đủ hơn.')
    return int(model.predict(x)[0]), cleaned


def compare_message(message, bundle):
    rows, neighbors = [], {}
    for name, (model, vectorizer) in bundle['configurations'].items():
        label, cleaned = predict_message(message, model, vectorizer)
        rows.append({'Cấu hình': name,
                     'Dự đoán': 'Spam / Lừa đảo' if label else 'Hợp lệ (Ham)'})
        if name.startswith('KNN'):
            distances, indices = model.kneighbors(vectorizer.transform([cleaned]))
            neighbors[name] = [
                {'row': int(i), 'message': bundle['messages'][i],
                 'label': int(bundle['labels'][i]), 'distance': float(distance)}
                for distance, i in zip(distances[0], indices[0])]
    return rows, neighbors
