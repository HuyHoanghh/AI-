from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TEXT_COLUMN = "message"
LABEL_COLUMN = "label"
HAM_LABEL = 0
SPAM_LABEL = 1

MAX_FEATURES = 5000
NGRAM_RANGE = (1, 1)
KNN_K_VALUES = [3, 5, 7, 9, 11, 15, 21]
CV_FOLDS = 5
RANDOM_STATE = 42

RESULT_COLUMNS = [
    "model", "feature", "accuracy", "precision", "recall", "f1",
    "training_time", "prediction_time"
]
