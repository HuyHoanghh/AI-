from pathlib import Path
import pandas as pd
from config.settings import TEXT_COLUMN, LABEL_COLUMN


def load_sms_csv(path: str | Path) -> pd.DataFrame:
    """Load one official Vietnamese SMS CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="raise").astype(int)

    invalid = set(df[LABEL_COLUMN].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Nhãn không hợp lệ: {invalid}")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "missing_message": int(df[TEXT_COLUMN].isna().sum()),
        "empty_message": int(df[TEXT_COLUMN].str.strip().eq("").sum()),
        "duplicates": int(df.duplicated(subset=[TEXT_COLUMN, LABEL_COLUMN]).sum()),
        "ham": int((df[LABEL_COLUMN] == 0).sum()),
        "spam": int((df[LABEL_COLUMN] == 1).sum()),
    }
