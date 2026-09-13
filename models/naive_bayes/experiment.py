"""Member 1 experiment: train-only CV, fixed official test, and audit artifacts."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from common.data_loader import load_sms_csv, dataset_summary
from common.eda import add_length_features, top_words, placeholder_counts
from common.feature_extraction import build_count_vectorizer, build_tfidf_vectorizer
from common.metrics import classification_metrics, timed_fit_predict, result_row
from common.preprocessing import clean_series
from config.settings import (
    TRAIN_PATH, TEST_PATH, PROJECT_ROOT, TEXT_COLUMN, LABEL_COLUMN,
    CV_FOLDS, RANDOM_STATE,
)
from models.naive_bayes.train_nb import build_naive_bayes

BUILDERS = {"Count": build_count_vectorizer, "TFIDF": build_tfidf_vectorizer}


def cross_validate_feature(text, labels, feature):
    # Vocabulary and IDF are learned independently inside every training fold.
    pipeline = Pipeline([
        ("vectorizer", BUILDERS[feature]()), ("model", build_naive_bayes()),
    ])
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, text, labels, cv=cv,
                            scoring=["accuracy", "precision", "recall", "f1"],
                            error_score="raise")
    return pd.DataFrame({"feature": feature, "fold": range(1, CV_FOLDS + 1),
                         **scores})


def error_table(test, cleaned, prediction):
    table = test.reset_index(drop=True).copy()
    table.insert(0, "test_row", range(len(table)))
    table["cleaned_message"] = list(cleaned)
    table["prediction"] = prediction
    table = table.loc[table[LABEL_COLUMN] != table["prediction"]].copy()
    table["error_type"] = table[LABEL_COLUMN].map({0: "FP", 1: "FN"})
    return table


def export_eda(train, test, train_path, test_path, out):
    summaries, lengths, words, tokens = [], [], [], []
    for split, df, path in [("train", train, train_path), ("test", test, test_path)]:
        summary = dataset_summary(df)
        # Count missing on raw input, before the common loader fills NaN.
        summary["missing_message"] = int(pd.read_csv(path)[TEXT_COLUMN].isna().sum())
        summaries.append({"split": split, **summary})
        features = add_length_features(df)
        stats = features.groupby(LABEL_COLUMN)[["char_count", "word_count"]].describe()
        stats.columns = ["_".join(c) for c in stats.columns]
        lengths.append(stats.reset_index().assign(split=split))
        tokens.append(placeholder_counts(df).assign(split=split))
        for label in (0, 1):
            words.extend({"split": split, "label": label, "word": w, "count": n}
                         for w, n in top_words(df, label))
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "eda_summary.csv", index=False)
    pd.concat(lengths).to_csv(out / "eda_lengths.csv", index=False)
    pd.concat(tokens).to_csv(out / "eda_placeholders.csv", index=False)
    pd.DataFrame(words).to_csv(out / "eda_top_words.csv", index=False)
    ax = summary.set_index("split")[["ham", "spam"]].plot.bar(rot=0)
    ax.set(xlabel="Official split", ylabel="Messages", title="Label distribution")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "eda_labels.png", dpi=150)
    plt.close(ax.figure)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (split, df) in zip(axes, [("train", train), ("test", test)]):
        for label, name in [(0, "Ham"), (1, "Spam")]:
            ax.hist(df.loc[df[LABEL_COLUMN] == label, TEXT_COLUMN].str.len(),
                    bins=30, alpha=.6, label=name)
        ax.set(title=split, xlabel="Message length (characters)", ylabel="Messages")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_lengths.png", dpi=150)
    plt.close(fig)
    return summary


def run_experiment(train_path=TRAIN_PATH, test_path=TEST_PATH, output_dir=None,
                   artifact_dir=None):
    out = Path(output_dir or PROJECT_ROOT / "results/naive_bayes")
    artifacts = Path(artifact_dir or PROJECT_ROOT / "saved_models")
    out.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    train = load_sms_csv(train_path)
    train_text = clean_series(train[TEXT_COLUMN])
    y_train = train[LABEL_COLUMN]
    folds = pd.concat([cross_validate_feature(train_text, y_train, f)
                       for f in BUILDERS], ignore_index=True)
    folds.to_csv(out / "cv_nb.csv", index=False)
    # Freeze the member 1 choice before reading the test file.
    cv_means = folds.groupby("feature")["test_f1"].mean()
    selected = cv_means.idxmax()
    test = load_sms_csv(test_path)
    test_text = clean_series(test[TEXT_COLUMN])
    summary = export_eda(train, test, train_path, test_path, out)
    rows, errors = [], []
    for feature, builder in BUILDERS.items():
        vectorizer = builder()
        x_train = vectorizer.fit_transform(train_text)
        x_test = vectorizer.transform(test_text)
        model = build_naive_bayes()
        pred, fit_time, pred_time = timed_fit_predict(model, x_train, y_train, x_test)
        metrics = classification_metrics(test[LABEL_COLUMN], pred)
        rows.append(result_row("NaiveBayes", feature, metrics, fit_time, pred_time))
        slug = feature.lower()
        cm = confusion_matrix(test[LABEL_COLUMN], pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
        display.plot(cmap="Blues", values_format="d", colorbar=False)
        display.ax_.set_title(f"Multinomial NB + {feature}")
        display.figure_.tight_layout()
        display.figure_.savefig(out / f"confusion_matrix_nb_{slug}.png", dpi=150)
        plt.close(display.figure_)
        table = error_table(test, test_text, pred)
        table.to_csv(out / f"errors_nb_{slug}.csv", index=False)
        errors.append(table.assign(feature=feature))
        joblib.dump(model, artifacts / f"nb_{slug}.joblib")
        joblib.dump(vectorizer, artifacts / f"vectorizer_{slug}.joblib")
    results = pd.DataFrame(rows)
    results.to_csv(out / "results_nb.csv", index=False)
    metadata = {
        "selected_nb_feature": selected, "selection_rule": "highest train-only 5-fold mean F1",
        "cv_mean_f1": cv_means.to_dict(), "alpha": 1.0, "random_state": RANDOM_STATE,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "timing": "seconds; model.fit and model.predict only; excludes vectorization and CV",
        "sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in [train_path, test_path]},
        "cleaned_train_test_overlap": len(set(train_text) & set(test_text)),
        "dataset_source": "https://github.com/trannguyenthaituan251209/vietnamese_sms_dataset",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Kết quả thực nghiệm người 1", "",
             "MultinomialNB alpha=1.0; dùng nguyên train/test chính thức. "
             "Count và TF-IDF dùng cấu hình chung, không fit vocabulary/IDF trên test.", "",
             "```text", summary.to_string(index=False), "```", "",
             "```text", results.to_string(index=False), "```", "",
             f"Cấu hình NB chọn bằng CV trên train: **{selected}**. "
             "Đây chưa phải mô hình tốt nhất toàn dự án; cần kết quả KNN.", "",
             "Thời gian tính bằng giây, chỉ đo model.fit/model.predict, không gồm vector hóa/CV.", "",
             f"Số nội dung trùng train/test sau làm sạch: {metadata['cleaned_train_test_overlap']}. "
             "Giữ nguyên split theo yêu cầu; cần cân nhắc khi diễn giải khả năng tổng quát hóa.", "",
             "## Ví dụ lỗi", "", "FP: Ham bị gán Spam. FN: Spam bị bỏ sót. "
             "Chọn 5 dòng đầu theo thứ tự test cho mỗi loại, mỗi cấu hình; không chọn theo xác suất."]
    for feature, table in zip(BUILDERS, errors):
        for kind in ["FP", "FN"]:
            subset = table.loc[table.error_type == kind]
            lines.extend(["", f"### {feature}: {kind} ({len(subset)} lỗi)", ""])
            if len(subset) < 5:
                lines.append("Có ít hơn 5 lỗi thực tế; liệt kê tất cả, không tạo thêm mẫu.")
            for row in subset.head(5).itertuples():
                message = str(row.message).replace("\n", " ").replace("\r", " ")
                lines.append(f"- Dòng test {row.test_row}: {message}")
    (out / "report_nb.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results
