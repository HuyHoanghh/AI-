from common.preprocessing import clean_message


def test_empty_message():
    assert clean_message(None) == ""
    assert clean_message("") == ""


def test_keeps_anonymized_tokens():
    text = clean_message("Giao dich [MONEY] luc [TIME], goi [PHONE]")
    assert "__money__" in text
    assert "__time__" in text
    assert "__phone__" in text


def test_normalizes_url():
    text = clean_message("Truy cap https://example.com ngay")
    assert "__url__" in text
