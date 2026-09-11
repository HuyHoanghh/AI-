import re


def clean_email(subject, message):
    """Normalize an Enron email into a single text string.

    Shared preprocessing used by both Naive Bayes and KNN pipelines.
    """
    subject = "" if subject is None else str(subject)
    message = "" if message is None else str(message)
    text = f"{subject} {message}".lower()

    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text)
    text = re.sub(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b", " EMAILTOKEN ", text)
    text = re.sub(r"\d+", " NUMTOKEN ", text)
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
