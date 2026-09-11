# Phân tích kết quả — Người 1

## Kết quả thực nghiệm

Dữ liệu chính thức gồm 2.394 mẫu train (1.754 Ham, 640 Spam), 597 mẫu test
(439 Ham, 158 Spam). Không có tin nhắn thiếu/rỗng. Có 303 dòng trùng
message+label trong train và 31 trong test; 31 nội dung sau làm sạch xuất hiện
ở cả train và test. Giữ nguyên split theo yêu cầu. CV chia theo dòng nên mẫu
trùng có thể nằm ở hai fold khác nhau; F1 CV có thể lạc quan.

| Cấu hình | Accuracy | Precision Spam | Recall Spam | F1 Spam | F1 CV train |
|---|---:|---:|---:|---:|---:|
| NB + Count | 85,43% | 69,19% | 81,01% | 74,64% | 82,36% |
| NB + TF-IDF | 87,44% | 81,68% | 67,72% | 74,05% | 78,00% |

Count được chọn từ CV train trước khi đọc test. Trên test, Count phát hiện
128/158 Spam nhưng chặn nhầm 57 Ham; TF-IDF phát hiện 107/158 Spam và chặn
nhầm 24 Ham. Count có Recall cao hơn, TF-IDF có Precision/Accuracy cao hơn.
F1 test hai cấu hình chỉ chênh khoảng 0,59 điểm phần trăm, chưa có kiểm định
để khẳng định khác biệt có ý nghĩa thống kê. Không thể kết luận hơn KNN
khi chưa có kết quả người 2.

## Phân tích 5 FP và 5 FN cho mỗi cấu hình

Nội dung nguyên mẫu và nhãn xem `results/naive_bayes/report_nb.md`; số dòng
bên dưới là chỉ số bắt đầu từ 0 của test.csv, không tính dòng tiêu đề.
Các diễn giải là giả thuyết từ việc đọc mẫu, không phải kết luận nhân quả
hay phép giải thích trọng số của mô hình. Nhãn được giữ theo dataset.

| Cấu hình/lỗi | Dòng | Nhận xét |
|---|---:|---|
| Count FP | 0 | Khuyến cáo y tế dài bị chặn; từ vựng ngoài chủ đề giao dịch thường gặp có thể gây nhầm. |
| Count FP | 3 | Thông báo an toàn giao thông chính thức bị nhầm; unigram không xác định được chủ thể gửi. |
| Count FP | 4 | Cảnh báo chống lừa đảo chứa “chuyển tiền”, “tài khoản”, link; mô hình khó hiểu phủ định “không”. |
| Count FP | 7 | Chúc mừng năm mới kèm khuyến cáo giao thông có cụm từ cũng gặp trong quảng cáo. |
| Count FP | 12 | Điện lực xác nhận khảo sát có số liên hệ, ngày và lời xưng hô dịch vụ; dễ lẫn tin mạo danh. |
| Count FN | 511 | Tin giả danh Viettel giống thông báo chuẩn hóa thuê bao hợp lệ, không có thông tin người gửi để phân biệt. |
| Count FN | 536 | Quảng bá cá cược không dấu, nhiều token tiền và URL; mất dấu làm khác từ vựng tiếng Việt có dấu. |
| Count FN | 541 | Quảng bá FLY88 dùng từ viết tắt, không dấu và ưu đãi; các từ có thể không đủ phân biệt. |
| Count FN | 543 | Quảng bá casino/World Cup chứa từ thương hiệu và token số; biến thể miền/từ vựng gây khó. |
| Count FN | 546 | Tin ngắn không dấu, miền nk99hv.com không có scheme nên không đổi thành __url__ bởi pipeline chung. |
| TFIDF FP | 4 | Cảnh báo chống lừa đảo bị hiểu giống nội dung lừa đảo do chia sẻ từ khóa. |
| TFIDF FP | 16 | Tin phòng chống đuối nước ngắn, ít ngữ cảnh và từ vựng chủ đề hiếm. |
| TFIDF FP | 22 | Hội thoại giao hàng viết tắt, thiếu dấu và dấu câu; token rời khó đại diện ngữ nghĩa. |
| TFIDF FP | 45 | Mã đăng nhập Telegram hợp lệ có link và số, cũng là dạng tín hiệu hay gặp trong tin giả danh. |
| TFIDF FP | 75 | Xác nhận đơn và gỡ khiếu nại có mã đơn, số giờ, lỗi chính tả; khó phân biệt bằng unigram. |
| TFIDF FN | 443 | Cảnh báo giao dịch giả danh VPBank rất giống thông báo ngân hàng thực. |
| TFIDF FN | 454 | Quảng cáo Shopee có từ thương hiệu, giảm giá và hướng dẫn đăng ký quen thuộc. |
| TFIDF FN | 456 | Quảng cáo FPT Shop cùng dạng mẫu; tên doanh nghiệp không chứng minh tính hợp lệ. |
| TFIDF FN | 459 | Quảng cáo spa đặt lịch gần với tin dịch vụ hợp lệ; ranh giới spam phụ thuộc cả ngữ cảnh đồng ý nhận. |
| TFIDF FN | 472 | Cảnh báo giả danh MB Bank có mẫu tương tự VPBank; nội dung đơn lẻ thiếu tín hiệu nguồn gửi. |

## Hạn chế và hướng phát triển

Naive Bayes giả định các đặc trưng độc lập có điều kiện, chưa hiểu phủ định
hoặc ngữ cảnh. Token ẩn danh bảo vệ thông tin cá nhân nhưng cũng làm tin
thật/giả giống nhau. Count/TF-IDF dùng unigram theo khoảng trắng, chưa tách
các từ ghép tiếng Việt. Dữ liệu mất cân bằng nên không chỉ nhìn Accuracy.

Các hướng có thể đánh giá ở vòng nghiên cứu riêng: CV theo nhóm nội dung
trùng, n-gram, chuẩn hóa miền không scheme, đánh giá tin không dấu và ngữ
cảnh người gửi. Mọi lựa chọn phải thực hiện trên train/validation và thống
nhất với người 2 trước khi sửa common/config. Không chỉnh pipeline dựa trên
các lỗi test này rồi coi kết quả chạy lại là đánh giá độc lập.

## Bàn giao

Đã có hai cấu hình, CV, metrics, thời gian, hai confusion matrix, EDA, bảng
lỗi, notebook và model/vectorizer riêng. Chưa tích hợp KNN, chưa chọn best
model toàn dự án, chưa push/merge GitHub. Tham khảo và lệnh tái lập nằm trong
`models/naive_bayes/README.md`.
