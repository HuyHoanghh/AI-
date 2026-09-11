from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "results" / "naive_bayes" / "results_nb.csv"
knn_path = ROOT / "results" / "knn" / "results_knn.csv"
out_path = ROOT / "results" / "final" / "final_comparison.csv"

frames = []
for path in (nb_path, knn_path):
    if path.exists():
        frames.append(pd.read_csv(path))

if not frames:
    raise SystemExit("No result CSV files found yet.")

final = pd.concat(frames, ignore_index=True)
if "f1" in final.columns:
    final = final.sort_values("f1", ascending=False)
final.to_csv(out_path, index=False)
print(final)
print(f"Saved: {out_path}")
