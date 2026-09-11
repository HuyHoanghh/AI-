# Ghi chú từ repo tham khảo

Repo tham khảo: https://github.com/Cham0703/DU_AN_SPAM

Những phần liên quan đã được rà soát:
- Load CSV và xử lý encoding.
- Xử lý `NaN` trước preprocessing.
- EDA độ dài tin nhắn theo nhãn.
- Thống kê top từ Spam/Ham và WordCloud.
- Vector hóa TF-IDF theo đúng nguyên tắc fit train / transform test.
- Đánh giá bằng classification metrics.
- Lưu vectorizer/model để tái sử dụng.

Các phần **không áp dụng nguyên xi**:
- Dataset tham khảo là SMS tiếng Anh và cột gốc `v1/v2`.
- English stopwords không phù hợp.
- Xóa toàn bộ số và ký tự đặc biệt có thể làm mất placeholder/đặc trưng quan trọng của bộ SMS tiếng Việt.
- `train_test_split` trong repo tham khảo được thay bằng train/test chính thức của Vietnamese SMS Dataset cho đánh giá cuối.
