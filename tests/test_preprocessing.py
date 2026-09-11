from common.preprocessing import clean_message


def test_empty_message():
    assert clean_message(None) == ""
    assert clean_message("") == ""


def test_keeps_anonymized_tokens():
    text = clean_message("Giao dịch [MONEY] lúc [TIME], gọi [PHONE], STK [BANK_ACC]")
    assert "__money__" in text
    assert "__time__" in text
    assert "__phone__" in text
    assert "__bank_acc__" in text


def test_normalizes_url_and_email():
    text = clean_message("Truy cập https://example.com hoặc gửi abc@example.com ngay")
    assert "__url__" in text
    assert "__email__" in text


def test_keeps_vietnamese_characters_and_digits():
    text = clean_message("Quý khách nhận 500000 đồng vào tài khoản")
    assert "quý" in text
    assert "khách" in text
    assert "500000" in text


def test_normalizes_whitespace_and_lowercase():
    text = clean_message("  CHÚC   MỪNG   QUÝ KHÁCH  ")
    assert text == "chúc mừng quý khách"
