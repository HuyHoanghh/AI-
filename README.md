# Enron Spam Detection

Machine-learning project for classifying Enron emails as **Spam** or **Ham** using two independent pipelines that can be developed in parallel and merged safely on GitHub.

## Models

- Member 1: Multinomial Naive Bayes
- Member 2: K-Nearest Neighbors (KNN)

Both pipelines must use the same shared preprocessing, feature settings, train/test split, and metrics.

## Git workflow

Recommended branches:

- `main`: stable release only
- `develop`: integration branch
- `feature/naive-bayes`: Member 1
- `feature/knn`: Member 2

Merge flow:

`feature/* -> develop -> main`

## Dataset

Use the Enron spam dataset chosen by the team. Do **not** commit the raw dataset to Git.

Place the local dataset at:

`data/raw/enron_spam.csv`

Expected core columns:

- `subject`
- `message`
- `label`

## Setup

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Shared configuration

See `config/settings.py` and `docs/experiment_rules.md`.

## Ownership

### Member 1

- `models/naive_bayes/`
- `notebooks/member1/`
- `results/naive_bayes/`
- `scripts/run_nb.py`

### Member 2

- `models/knn/`
- `notebooks/member2/`
- `results/knn/`
- `scripts/run_knn.py`

### Shared / protected by agreement

- `common/`
- `config/`
- `README.md`
- `scripts/compare_models.py`
- `app/`

## Final comparison

After both model result CSV files exist:

```bash
python scripts/compare_models.py
```

The merged result will be written to:

`results/final/final_comparison.csv`
