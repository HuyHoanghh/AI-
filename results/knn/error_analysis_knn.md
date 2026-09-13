# Error Analysis — KNN (Người 2)

Nguồn lỗi được trích từ `test.csv` (597 mẫu). Mỗi cấu hình liệt kê tối đa
5 False Positive (Ham bị đoán thành Spam) và 5 False Negative (Spam bị đoán thành Ham).

## KNN + Count (best k = 3)

- Accuracy = 0.9397, Precision = 0.8631, Recall = 0.9177, F1 = 0.8896
- Tổng số FP = 23, tổng số FN = 13

### False Positives (Ham → đoán Spam)

- Công ty điện lực thủ đức đã tiếp nhận thông tin của quý khách ! ĐL đã cử a Hoàng khảo sát với số ĐT :[NUMBER] khảo sát ngày [DATE]
- Mung sinh nhat 15 tuoi Lotte Cinema tang ban 2 code xem phim [MONEY] chi voi [MONEY].Dung tai rap tu 10-13/4,17-20/4,24-27/4. Code: 1602300002820093 va 1602300003125533.
- Ahamove tang rieng ban ma HIAHA tiet kiem ngay [MONEY] cho 2 don hang dau tien, dat don ngay tai https://ahamove.onelink.me/fmLb/5huavlai
- The Manson thong bao, don hang: 221007R4FP4BRJ da duoc kich hoat bao hanh dien tu. Chuc quy khach co giay phut trai nghiem that tot san pham. Chi tiet lien he: https://themanson.vn/
- Tôi đã nhận được đơn hàng spxvn[NUMBER]a  Vào lúc 13h10 ngày 6 thang 10 . Đồng ý gở khieu nai

### False Negatives (Spam → đoán Ham)

- [QC]QK du dieu kien vay tieu dung den [MONEY] VND, ls uu dai tu EASY CREDIT (EVN Finance). Soan Y gui [NUMBER] de DK vay va dong y cho MobiFone chia se tt thue bao de ho tro tham dinh khoan vay.Tu cho…
- TT: [NUMBER].G@me bài tra thu0ng kie'mtien cuc de.ta?i luon: bitly.com.vn nhan 88 kthu nghiem nhieu tr0 hap dan..rut nap sieu t0c
- CANH CAO LAN CUOI Can Cu Theo BO HINH SU Khoan 1, Dieu 139-141 dua tren Quyet Dinh [NUMBER]/[NUMBER]-TTCP "HANH VI LUA DAO CHIEM DOAT TAI SAN" Doi tuong [NAME] [DATE]. CO TINH TRI HOAN keo dai thoi gi…
- AN NINH NGAN HANG Thong bao: Can cu vao hop dong tra gop cua Ong/Ba: [NAME] Sinh ngay [DATE] co tinh khong hop tac. Ho so Ong/Ba chinh thuc bi lap bien ban thanh ly truoc han va ban giao cho BO [NAME]…
- Qua cong tac dieu tra tim kiem tai dia phuong chung toi da co du dieu kien va co so de nho Cong An vao cuoc ho tro dieu tra trong qua trinh lien he Ong/Ba [NAME] [DATE] va nguoi than lien tuc/tat nguo…

## KNN + TFIDF (best k = 3)

- Accuracy = 0.9296, Precision = 0.8537, Recall = 0.8861, F1 = 0.8696
- Tổng số FP = 24, tổng số FN = 18

### False Positives (Ham → đoán Spam)

- đơn 51/9 đường 475 a giao giùm e [TIME] đc kh a  Có đon od em bỏ vô nhà có j lây giúp em nha a tài  dạ oke e cảm ơn a
- (TB) VNPT VinaPhone tran trong thong bao chuong trinh lap cap quang Wifi mien phi hoa mang tai TP.HCM. Chi tiet xin vui long LH:[NUMBER]
- Vui long nhap mat khau [NUMBER] de dang nhap. Tran trong!
- Tầm [TIME] anh giao lại giùm em được không anh h nhà em khôg có ng nhận anh ơi  Ok. Có người nhà a hẹn [TIME] ạ
- Em ơi, sáng giờ gọi e 2 lần để giao hàng r. C bên Dr Mai trị mụn

### False Negatives (Spam → đoán Ham)

- FLY88 trai nghiem moi: Dang ky nhan [MONEY]. Tai app nhan [MONEY]. X2 nap dau trai nghiem. Diem danh moi ngay nhan [MONEY]. Dang ky: https://fly88.health
- Vao ngay WW88 vui cung WORLD CUP Nap [MONEY] nhan [MONEY] dang cap the thao, nohu, casino tai https://ww98dt3.cc/Register? f=[NUMBER]
- DANG KY TAI KHOAN NAP DAU 50 NHAN 50 nk99hv.com
- [QC]QK du dieu kien vay tieu dung den [MONEY] VND, ls uu dai tu EASY CREDIT (EVN Finance). Soan Y gui [NUMBER] de DK vay va dong y cho MobiFone chia se tt thue bao de ho tro tham dinh khoan vay.Tu cho…
- Anh Chi Oi: (rKYVFnOi). E Nhan [NAME] l/Xe. CMND. Can Cuoc. Bang Dai Hoc. THPT. Cao Dang. Nhan Hang thu Tien. E: [NUMBER]. Xin cam on!

## Nhận xét chung

- **False Positive** thường là tin nhắn Ham mang phong cách quảng bá/thông báo
  (chứa `[MONEY]`, ưu đãi, lời mời) nên láng giềng của chúng lẫn nhiều Spam.
- **False Negative** thường là Spam ngắn, ít từ vựng đặc trưng hoặc trùng lặp
  nội dung với tin Ham, khiến đa số láng giềng bỏ phiếu Ham.
- Với dữ liệu SMS thưa và mất cân bằng nhẹ, KNN + CountVectorizer cho F1 cao hơn
  một chút so với TF-IDF vì khoảng cách cosine trên vector đếm giữ được tín hiệu
  token xuất hiện lặp lại (đặc trưng của tin quảng cáo/lừa đảo).
