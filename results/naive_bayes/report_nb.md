# Kết quả thực nghiệm người 1

MultinomialNB alpha=1.0; dùng nguyên train/test chính thức. Count và TF-IDF dùng cấu hình chung, không fit vocabulary/IDF trên test.

```text
split  rows  missing_message  empty_message  duplicates  ham  spam
train  2394                0              0         303 1754   640
 test   597                0              0          31  439   158
```

```text
     model feature  accuracy  precision   recall       f1  training_time  prediction_time
NaiveBayes   Count  0.854271   0.691892 0.810127 0.746356       0.003897         0.000508
NaiveBayes   TFIDF  0.874372   0.816794 0.677215 0.740484       0.003692         0.000516
```

Cấu hình NB chọn bằng CV trên train: **Count**. Đây chưa phải mô hình tốt nhất toàn dự án; cần kết quả KNN.

Thời gian tính bằng giây, chỉ đo model.fit/model.predict, không gồm vector hóa/CV.

Số nội dung trùng train/test sau làm sạch: 31. Giữ nguyên split theo yêu cầu; cần cân nhắc khi diễn giải khả năng tổng quát hóa.

## Ví dụ lỗi

FP: Ham bị gán Spam. FN: Spam bị bỏ sót. Chọn 5 dòng đầu theo thứ tự test cho mỗi loại, mỗi cấu hình; không chọn theo xác suất.

### Count: FP (57 lỗi)

- Dòng test 0: Để phòng ngừa bệnh tay chân miệng, người chăm sóc trẻ và trẻ cần rửa tay thường xuyên bằng xà phòng; Lau sạch các bề mặt, vật dụng tiếp xúc hàng ngày như đồ chơi, bàn ghế, sàn nhà... Đưa trẻ đi khám ngay tại các cơ sở y tế gần nhất khi trẻ có các biểu hiện sốt, nổi hồng ban bóng nước ở lòng bàn tay, bàn chân, loét miệng, giật mình.
- Dòng test 3: (TB) Phòng Cảnh sát giao thông đường bộ - đường sắt Công an Thành phố Hồ Chí Minh khuyến cáo: điều khiển phương tiện sau khi đã sử dụng rượu bia là một trong những nguyên nhân hàng đầu dẫn đến các vụ tai nạn giao thông đặc biệt nghiêm trọng, để lại hậu quả thương tâm cho chính gia đình bạn và xã hội. Vì sức khỏe và an toàn của chính bạn: “Đã uống rượu bia - không lái xe”.
- Dòng test 4: (TB) Phòng Cảnh sát giao thông đường bộ - đường sắt Công an Thành phố Hồ Chí Minh khuyến cáo: Cảnh sát giao thông không gọi điện thoại để yêu cầu chuyển tiền nộp phạt vi phạm hành chính qua các link hoặc tài khoản do các đối tượng lừa đảo cung cấp. Việc nộp phạt vi phạm hành chính được thực hiện trực tiếp tại trụ sở lực lượng Cảnh sát giao thông hoặc thông qua Cổng dịch vụ công quốc gia tại địa chỉ https://dichvucong.gov.vn hoặc Cổng dịch vụ công Bộ Công an tại địa chỉ https://dichvucong.bocongan.gov.vn
- Dòng test 7: (TB) Chúc mừng năm mới Quý Mão [NUMBER]! Người dân chủ động sắp xếp thời gian trở lại nơi làm việc và sinh sống tại các thành phố lớn, tránh tình trạng ùn tắc giao thông vào cuối dịp nghỉ lễ, tuyệt đối không điều khiển phương tiện khi đã sử dụng rượu, bia và các chất kích thích khác.
- Dòng test 12: Công ty điện lực thủ đức đã tiếp nhận thông tin của quý khách ! ĐL đã cử a Hoàng khảo sát với số ĐT :[NUMBER] khảo sát ngày [DATE]

### Count: FN (30 lỗi)

- Dòng test 511: [Viettel] Thuê bao của quý khách chưa chuẩn hóa thông tin theo quy định, sẽ bị khóa 1 chiều sau [TIME]. Vui lòng gọi [NUMBER] để cập nhật.
- Dòng test 536: Hoi vien moi nap dau [MONEY] tang [MONEY] tao tai khoan moi nhan uu dai lixi moi https://www.ee8qbet.com/?ch=0b8857e136
- Dòng test 541: FLY88 trai nghiem moi: Dang ky nhan [MONEY]. Tai app nhan [MONEY]. X2 nap dau trai nghiem. Diem danh moi ngay nhan [MONEY]. Dang ky: https://fly88.health
- Dòng test 543: Vao ngay WW88 vui cung WORLD CUP Nap [MONEY] nhan [MONEY] dang cap the thao, nohu, casino tai https://ww98dt3.cc/Register? f=[NUMBER]
- Dòng test 546: DANG KY TAI KHOAN NAP DAU 50 NHAN 50 nk99hv.com

### TFIDF: FP (24 lỗi)

- Dòng test 4: (TB) Phòng Cảnh sát giao thông đường bộ - đường sắt Công an Thành phố Hồ Chí Minh khuyến cáo: Cảnh sát giao thông không gọi điện thoại để yêu cầu chuyển tiền nộp phạt vi phạm hành chính qua các link hoặc tài khoản do các đối tượng lừa đảo cung cấp. Việc nộp phạt vi phạm hành chính được thực hiện trực tiếp tại trụ sở lực lượng Cảnh sát giao thông hoặc thông qua Cổng dịch vụ công quốc gia tại địa chỉ https://dichvucong.gov.vn hoặc Cổng dịch vụ công Bộ Công an tại địa chỉ https://dichvucong.bocongan.gov.vn
- Dòng test 16: Trẻ em cần được học bơi và học kỹ năng an toàn để phòng, chống đuối nước.
- Dòng test 22: đơn 51/9 đường 475 a giao giùm e [TIME] đc kh a  Có đon od em bỏ vô nhà có j lây giúp em nha a tài  dạ oke e cảm ơn a
- Dòng test 45: Telegram code: [NUMBER]   You can also tap on this link to log in:   https://t.me/login/[NUMBER]
- Dòng test 75: Tôi đã nhận được đơn hàng spxvn[NUMBER]a  Vào lúc 13h10 ngày 6 thang 10 . Đồng ý gở khieu nai

### TFIDF: FN (51 lỗi)

- Dòng test 443: VPBank thông báo: thẻ của quý khách vừa phát sinh giao dịch [MONEY] lúc [TIME]. Nếu không phải quý khách, gọi ngay [NUMBER] để khóa thẻ.
- Dòng test 454: Shopee SALE lớn: giảm tới 50% toàn bộ sản phẩm, freeship toàn quốc. Soạn DK gửi 8x18 để nhận mã giảm giá.
- Dòng test 456: FPT Shop SALE lớn: giảm tới 50% toàn bộ sản phẩm, freeship toàn quốc. Soạn DK gửi 8x28 để nhận mã giảm giá.
- Dòng test 459: Spa Hà ưu đãi 70% gói chăm sóc da dịp lễ, tặng kèm 2 buổi trị liệu. Gọi [NUMBER] đặt lịch ngay hôm nay.
- Dòng test 472: MB Bank thông báo: thẻ của quý khách vừa phát sinh giao dịch [MONEY] lúc [TIME]. Nếu không phải quý khách, gọi ngay [NUMBER] để khóa thẻ.
