import re
import unicodedata

# Token do bộ dữ liệu đã ẩn danh PII. Ta bảo toàn chúng như đặc trưng.
PLACEHOLDER_RE = re.compile(r"\[(PHONE|BANK_ACC|MONEY|NUMBER|TIME|DATE)\]", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)
SPACE_RE = re.compile(r"\s+")


def clean_message(message) -> str:
    """Preprocess Vietnamese SMS without destroying useful fraud indicators."""
    if message is None:
        return ""
    text = str(message).strip()
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)

    # Protect official anonymization tokens before lowercasing.
    text = PLACEHOLDER_RE.sub(lambda m: f" __{m.group(1).upper()}__ ", text)
    text = URL_RE.sub(" __URL__ ", text)
    text = EMAIL_RE.sub(" __EMAIL__ ", text)
    text = text.lower()

    # Keep Vietnamese letters, digits, underscores and whitespace.
    # Digits can be useful in spam; we do not delete them globally.
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def clean_series(series):
    return series.fillna("").map(clean_message)
