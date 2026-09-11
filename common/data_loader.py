import pandas as pd

from config.settings import SUBJECT_COLUMN, MESSAGE_COLUMN, TEXT_COLUMN
from common.preprocessing import clean_email


def load_csv(path):
    df = pd.read_csv(path)
    if SUBJECT_COLUMN not in df.columns:
        df[SUBJECT_COLUMN] = ""
    if MESSAGE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {MESSAGE_COLUMN}")
    df[TEXT_COLUMN] = [
        clean_email(s, m)
        for s, m in zip(df[SUBJECT_COLUMN], df[MESSAGE_COLUMN])
    ]
    return df
