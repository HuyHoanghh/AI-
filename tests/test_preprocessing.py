from common.preprocessing import clean_email


def test_clean_email_replaces_tokens():
    text = clean_email(
        "WIN 1000",
        "Click https://example.com or mail test@example.com",
    )
    assert "urltoken" in text.lower()
    assert "emailtoken" in text.lower()
    assert "numtoken" in text.lower()
