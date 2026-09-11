from collections import Counter
import pandas as pd
from config.settings import TEXT_COLUMN, LABEL_COLUMN
from common.preprocessing import clean_series


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["char_count"] = out[TEXT_COLUMN].fillna("").str.len()
    out["word_count"] = out[TEXT_COLUMN].fillna("").str.split().str.len()
    return out


def top_words(df: pd.DataFrame, label: int, n: int = 20):
    cleaned = clean_series(df.loc[df[LABEL_COLUMN] == label, TEXT_COLUMN])
    words = " ".join(cleaned).split()
    return Counter(words).most_common(n)


def placeholder_counts(df: pd.DataFrame) -> pd.DataFrame:
    tokens = ["[PHONE]", "[BANK_ACC]", "[MONEY]", "[NUMBER]", "[TIME]", "[DATE]"]
    rows = []
    for label in (0, 1):
        subset = df.loc[df[LABEL_COLUMN] == label, TEXT_COLUMN].fillna("")
        row = {"label": label}
        for token in tokens:
            row[token] = int(subset.str.contains(token, regex=False, case=False).sum())
        rows.append(row)
    return pd.DataFrame(rows)
