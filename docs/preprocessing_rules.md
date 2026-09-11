# Quy tắc preprocessing chung

Hai thành viên phải dùng cùng hàm `common.preprocessing.clean_message`.

## Quy tắc
1. Xử lý `None`/chuỗi rỗng an toàn.
2. Unicode NFC.
3. Giữ các placeholder chính thức: `[PHONE]`, `[BANK_ACC]`, `[MONEY]`, `[NUMBER]`, `[TIME]`, `[DATE]` dưới dạng token.
4. URL -> `__URL__`, email address -> `__EMAIL__`.
5. Lowercase.
6. Chuẩn hóa khoảng trắng.
7. Không xóa toàn bộ chữ số vì số có thể mang tín hiệu spam/lừa đảo.

## Khác với repo tham khảo DU_AN_SPAM
Repo tham khảo xử lý SMS tiếng Anh bằng lowercase, xóa số, xóa ký tự đặc biệt và English stopwords. Ta chỉ tham khảo cấu trúc xử lý và EDA; với SMS tiếng Việt hiện tại, xóa số/token ẩn danh có thể làm mất tín hiệu nên pipeline được điều chỉnh.
