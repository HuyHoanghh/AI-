import joblib
import streamlit as st
from pathlib import Path
from common.preprocessing import clean_message

st.set_page_config(page_title="Vietnamese SMS Spam Detection")
st.title("Phát hiện tin nhắn rác tiếng Việt")
st.caption("Demo sẽ dùng model tốt nhất sau giai đoạn so sánh.")

model_path = Path("saved_models/best_model.joblib")
vectorizer_path = Path("saved_models/best_vectorizer.joblib")
message = st.text_area("Nhập nội dung SMS")

if st.button("Phân loại"):
    if not model_path.exists() or not vectorizer_path.exists():
        st.warning("Chưa có best_model.joblib và best_vectorizer.joblib. Hãy hoàn thành thí nghiệm trước.")
    elif not message.strip():
        st.info("Vui lòng nhập tin nhắn.")
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        x = vectorizer.transform([clean_message(message)])
        pred = int(model.predict(x)[0])
        if pred == 1:
            st.error("Kết quả: SPAM / LỪA ĐẢO")
        else:
            st.success("Kết quả: HỢP LỆ")
