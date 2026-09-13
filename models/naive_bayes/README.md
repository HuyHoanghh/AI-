# Phần người 1 — Multinomial Naive Bayes

Đọc yêu cầu tại `PRODUCT_REQUIREMENTS.md` và `docs/TEAM_CHECKLIST.md`.

## Chạy

Từ thư mục gốc, sau khi cài `requirements.txt` và tải dữ liệu:

```powershell
python -m scripts.download_dataset
python scripts/run_nb.py
python -m pytest -q
```

Máy hiện tại đã có Python cục bộ, có thể dùng:

```powershell
D:/ttnhantao/python-runtime/python.exe D:/ttnhantao/AI/scripts/run_nb.py
```

Notebook: `notebooks/member1/naive_bayes_experiment.ipynb`.
Để chạy notebook trên máy khác: `python -m pip install nbformat nbclient ipykernel`.

## Thiết kế

- Dùng nguyên train/test chính thức và `common.preprocessing.clean_series`.
- MultinomialNB với alpha=1.0, unigram, tối đa 5.000 đặc trưng theo config chung.
- So sánh Count và TF-IDF bằng StratifiedKFold 5 fold, seed=42 trên train.
- Vectorizer nằm trong Pipeline của CV, được fit riêng trong mỗi fold.
- Chọn cấu hình NB bằng F1 trung bình CV trước khi đọc test; test chỉ đánh giá cuối.
- Precision/Recall/F1 dành cho lớp Spam (1). Thời gian giây chỉ gồm model.fit/predict.
- Không sửa common/config và không triển khai phần KNN của người 2.

## Đầu ra

`results/naive_bayes/` gồm CSV metrics, CV từng fold, hai confusion matrix,
bảng EDA, hai biểu đồ EDA, toàn bộ lỗi FP/FN, metadata SHA256 và `report_nb.md`.
`docs/member1_analysis.md` giải thích kết quả và các lỗi thực tế.
Model/vectorizer từng cấu hình được lưu ở `saved_models/` theo tên hiện có.
Chưa ghi `best_model.joblib`: lựa chọn cuối toàn dự án cần tích hợp kết quả KNN.

## Tham khảo

Đã đọc `final_email_detechtion.ipynb` tại https://github.com/Cham0703/DU_AN_SPAM
(commit e1f6b37bc92287c4b81954fe0661751df1d20b7f).
Tham khảo cách thống kê độ dài, tần suất từ, metrics, confusion matrix và lưu artifact.
Viết lại cho tiếng Việt; không dùng stopwords tiếng Anh hay xóa hết chữ số.
Notebook tham khảo có đoạn fit Count/TF-IDF trên toàn bộ dữ liệu trước CV/chia tập;
thực nghiệm này tránh cách làm đó bằng Pipeline trong từng fold.

Dữ liệu: https://github.com/trannguyenthaituan251209/vietnamese_sms_dataset
(CC BY 4.0; nhóm Trần Nguyễn Thái Tuấn và cộng sự). Dùng train.csv/test.csv,
không dùng spam.csv tiếng Anh của kho tham khảo.
