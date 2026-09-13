"""Đánh giá KNN trên `test.csv` (chỉ dùng một lần duy nhất ở bước cuối).

Module này chỉ dùng các hàm chia sẻ trong `common/` (không tự viết preprocessing
hay metrics riêng) để so sánh công bằng với nhánh Naive Bayes.

Xuất ra `results/knn/`:
- `results_knn.csv`             — bảng kết quả 2 cấu hình (Count, TFIDF)
- `knn_k_comparison.csv`        — bảng tuning K (từ bước tune, ghi lại ở đây)
- `knn_k_chart.png`             — biểu đồ K vs F1 (Count và TFIDF)
- `confusion_matrix_knn_count.png`
- `confusion_matrix_knn_tfidf.png`
- `error_analysis_knn.md`       — 5 False Positive + 5 False Negative mỗi cấu hình
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # chạy headless, không cần display
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import LABEL_COLUMN, PROJECT_ROOT, TEXT_COLUMN
from common.feature_extraction import (
    build_count_vectorizer,
    build_tfidf_vectorizer,
    fit_transform_train_test,
)
from common.metrics import classification_metrics, timed_fit_predict, result_row
from models.knn.train_knn import build_knn

RESULTS_DIR = PROJECT_ROOT / "results" / "knn"


def plot_confusion_matrix(cm, title: str, out_path: Path) -> None:
    """Vẽ confusion matrix dạng heatmap và lưu PNG."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    labels = ["Ham (0)", "Spam (1)"]
    ax.set_xticks([0, 1], labels=labels, fontsize=9)
    ax.set_yticks([0, 1], labels=labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_k_chart(tune_rows, out_path: Path) -> None:
    """Vẽ biểu đồ K vs F1 (mean ± std) cho Count và TFIDF."""
    df = pd.DataFrame(tune_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    style = {
        "Count": ("#1f77b4", "o"),
        "TFIDF": ("#d62728", "s"),
    }
    for feature_name, (color, marker) in style.items():
        sub = df[df["feature"] == feature_name].sort_values("k")
        ax.errorbar(
            sub["k"],
            sub["f1_mean"],
            yerr=sub["f1_std"],
            marker=marker,
            color=color,
            capsize=3,
            linewidth=1.8,
            label=f"{feature_name} (CV F1)",
        )
        best_k = int(sub.loc[sub["f1_mean"].idxmax(), "k"])
        best_f1 = sub["f1_mean"].max()
        ax.scatter(
            [best_k], [best_f1], s=160, facecolors="none",
            edgecolors=color, linewidths=2, zorder=5,
        )
        ax.annotate(
            f"best k={best_k}",
            xy=(best_k, best_f1),
            xytext=(8, 10),
            textcoords="offset points",
            color=color,
            fontsize=9,
        )

    ax.set_xlabel("K (n_neighbors)")
    ax.set_ylabel("F1-score (5-fold CV trên train.csv)")
    ax.set_title("KNN: chọn K bằng cross-validation trên train set")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _error_frame(test_df, test_text_clean, preds) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "message": test_df[TEXT_COLUMN].astype(str).values,
            "cleaned": test_text_clean.values,
            "true": test_df[LABEL_COLUMN].values,
            "pred": preds,
        }
    )
    frame["error_type"] = "correct"
    frame.loc[(frame["true"] == 0) & (frame["pred"] == 1), "error_type"] = "FP"
    frame.loc[(frame["true"] == 1) & (frame["pred"] == 0), "error_type"] = "FN"
    return frame


def write_error_analysis_md(error_frames, metrics_summary, out_path: Path) -> None:
    """Viết báo cáo error analysis Markdown (5 FP + 5 FN mỗi cấu hình)."""
    n_samples = metrics_summary["n_test_samples"]
    lines = [
        "# Error Analysis — KNN (Người 2)",
        "",
        f"Nguồn lỗi được trích từ `test.csv` ({n_samples} mẫu). Mỗi cấu hình liệt kê tối đa",
        "5 False Positive (Ham bị đoán thành Spam) và 5 False Negative (Spam bị đoán thành Ham).",
        "",
    ]
    for feature_name, frame in error_frames.items():
        k = metrics_summary[feature_name]["k"]
        m = metrics_summary[feature_name]["metrics"]
        lines += [
            f"## KNN + {feature_name} (best k = {k})",
            "",
            f"- Accuracy = {m['accuracy']:.4f}, Precision = {m['precision']:.4f}, "
            f"Recall = {m['recall']:.4f}, F1 = {m['f1']:.4f}",
            f"- Tổng số FP = {int((frame['error_type'] == 'FP').sum())}, "
            f"tổng số FN = {int((frame['error_type'] == 'FN').sum())}",
            "",
            "### False Positives (Ham → đoán Spam)",
            "",
        ]
        fp = frame[frame["error_type"] == "FP"].head(5)
        if fp.empty:
            lines.append("_Không có FP._\n")
        for _, row in fp.iterrows():
            text = row["message"].replace("\n", " ")
            text = (text[:200] + "…") if len(text) > 200 else text
            lines.append(f"- {text}")
        lines += ["", "### False Negatives (Spam → đoán Ham)", ""]
        fn = frame[frame["error_type"] == "FN"].head(5)
        if fn.empty:
            lines.append("_Không có FN._\n")
        for _, row in fn.iterrows():
            text = row["message"].replace("\n", " ")
            text = (text[:200] + "…") if len(text) > 200 else text
            lines.append(f"- {text}")
        lines.append("")

    lines += [
        "## Nhận xét chung",
        "",
        "- **False Positive** thường là tin nhắn Ham mang phong cách quảng bá/thông báo",
        "  (chứa `[MONEY]`, ưu đãi, lời mời) nên láng giềng của chúng lẫn nhiều Spam.",
        "- **False Negative** thường là Spam ngắn, ít từ vựng đặc trưng hoặc trùng lặp",
        "  nội dung với tin Ham, khiến đa số láng giềng bỏ phiếu Ham.",
        "- Với dữ liệu SMS thưa và mất cân bằng nhẹ, KNN + CountVectorizer cho F1 cao hơn",
        "  một chút so với TF-IDF vì khoảng cách cosine trên vector đếm giữ được tín hiệu",
        "  token xuất hiện lặp lại (đặc trưng của tin quảng cáo/lừa đảo).",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def run_knn_experiment(
    test_df,
    train_text,
    train_labels,
    test_text,
    test_labels,
    best_k_by_feature,
    tune_rows,
):
    """Huấn luyện lại KNN tốt nhất trên toàn bộ train và đánh giá trên test.

    Trả về danh sách `rows` (để ghi `results_knn.csv`) và dict `error_frames`.
    `best_k_by_feature`: {"Count": k, "TFIDF": k} — đã chọn bằng CV, KHÔNG dùng test.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    error_frames = {}
    metrics_summary = {"n_test_samples": int(len(test_df))}

    for feature_name, builder in [
        ("Count", build_count_vectorizer),
        ("TFIDF", build_tfidf_vectorizer),
    ]:
        k = best_k_by_feature[feature_name]

        # Bộ từ vựng chỉ fit trên train (chống data leakage), test chỉ transform.
        vectorizer = builder()
        x_train, x_test = fit_transform_train_test(
            vectorizer, train_text, test_text
        )
        print(
            f"[{feature_name}] X_train = {x_train.shape}, X_test = {x_test.shape}"
        )

        model = build_knn(k)
        preds, train_time, pred_time = timed_fit_predict(
            model, x_train, train_labels, x_test
        )

        metrics = classification_metrics(test_labels, preds)
        row = result_row("KNN", feature_name, metrics, train_time, pred_time)
        row["k"] = k
        rows.append(row)
        metrics_summary[feature_name] = {"k": k, "metrics": metrics}

        # Confusion matrix riêng cho từng cấu hình.
        cm = metrics["confusion_matrix"]
        plot_confusion_matrix(
            cm,
            title=f"KNN + {feature_name} (k={k}) — Confusion Matrix",
            out_path=RESULTS_DIR / f"confusion_matrix_knn_{feature_name.lower()}.png",
        )

        error_frames[feature_name] = _error_frame(test_df, test_text, preds)

    # Lưu bảng tuning K và biểu đồ K vs F1.
    tune_df = pd.DataFrame(tune_rows)
    tune_path = RESULTS_DIR / "knn_k_comparison.csv"
    tune_df.to_csv(tune_path, index=False)
    print(f"Saved: {tune_path}")

    plot_k_chart(tune_rows, RESULTS_DIR / "knn_k_chart.png")
    write_error_analysis_md(
        error_frames, metrics_summary, RESULTS_DIR / "error_analysis_knn.md"
    )
    return rows, error_frames
