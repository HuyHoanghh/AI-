from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config.settings import PROJECT_ROOT


def main():
    nb_path = PROJECT_ROOT / "results" / "naive_bayes" / "results_nb.csv"
    knn_path = PROJECT_ROOT / "results" / "knn" / "results_knn.csv"
    nb = pd.read_csv(nb_path)
    knn = pd.read_csv(knn_path)
    final = pd.concat([nb, knn], ignore_index=True).sort_values("f1", ascending=False)
    out = PROJECT_ROOT / "results" / "final" / "final_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(out, index=False)
    print(final.to_string(index=False))
    print(f"\nBest configuration: {final.iloc[0]['model']} + {final.iloc[0]['feature']}")


if __name__ == "__main__":
    main()
