"""Run the reproducible member 1 experiment from any working directory."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.naive_bayes.experiment import run_experiment


def main():
    print(run_experiment().to_string(index=False))


if __name__ == "__main__":
    main()
