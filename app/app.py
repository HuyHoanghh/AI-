from pathlib import Path
import sys
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import joblib
from app.inference import ARTIFACT_DIR, load_artifacts, predict_message
from app.inference import compare_message
from common.data_loader import load_sms_csv
from config.settings import TEST_PATH

st.set_page_config(page_title='Phân loại SMS tiếng Việt', page_icon='✉️', layout='centered')
st.title('Phân loại SMS tiếng Việt')
st.write('Nhập một tin nhắn để nhận dự đoán **Hợp lệ** hoặc **Spam / Lừa đảo**.')


@st.cache_resource
def cached_artifacts(model_stamp, vectorizer_stamp):
    return load_artifacts()


@st.cache_resource
def cached_comparison(path, stamp):
    return joblib.load(path)


try:
    stamps = [(ARTIFACT_DIR / name).stat().st_mtime_ns for name in
              ('best_model.joblib', 'best_vectorizer.joblib')]
    model, vectorizer = cached_artifacts(*stamps)
except FileNotFoundError:
    st.error('Demo chưa có mô hình. Chạy lệnh chuẩn bị dưới đây rồi tải lại trang.')
    st.code('python scripts/export_demo_model.py')
    st.stop()
except Exception:
    st.error('Không tải được mô hình. Hãy chạy lại scripts/export_demo_model.py trong môi trường hiện tại.')
    st.stop()

st.caption('KNN · CountVectorizer · K = 3')


def set_example(text):
    st.session_state['sms'] = text
    st.session_state.pop('random_sample', None)


@st.cache_data
def load_test_samples(stamp):
    data = load_sms_csv(TEST_PATH)
    return data.loc[data.message.str.strip().ne('')].to_dict('records')


def random_example(samples):
    previous = st.session_state.get('sms', '')
    choices = [sample for sample in samples if sample['message'] != previous]
    sample = random.choice(choices or samples)
    st.session_state['sms'] = sample['message']
    st.session_state['random_sample'] = sample


try:
    samples = load_test_samples(TEST_PATH.stat().st_mtime_ns)
except (OSError, ValueError):
    samples = []
st.button('🎲 Lấy tin nhắn ngẫu nhiên', key='random_sms',
          on_click=random_example, args=(samples,), disabled=not samples)
if samples:
    st.caption('Lấy từ tập test có sẵn. Bấm lại để thử nội dung khác, rồi bấm Phân loại.')
else:
    st.caption('Chưa có dữ liệu mẫu. Hãy tải data/raw/test.csv để dùng nút ngẫu nhiên.')


with st.expander('Thử nhanh với tin nhắn minh họa'):
    st.caption('Các mẫu tự viết để thao tác thử, không phải kết quả kiểm thử độc lập.')
    c1, c2 = st.columns(2)
    c1.button('Tin nhắn hẹn gặp', on_click=set_example,
              args=('Chiều nay mình gặp nhau ở quán cà phê nhé.',))
    c2.button('Tin nhắn quảng cáo', on_click=set_example,
              args=('Đăng ký tài khoản nhận thưởng [MONEY]. Nạp tiền ngay để nhận ưu đãi tại https://example.com',))

with st.form('classify'):
    message = st.text_area('Nội dung tin nhắn', key='sms', height=180, max_chars=5000,
                           placeholder='Dán hoặc nhập nội dung SMS tại đây…')
    submitted = st.form_submit_button('Phân loại', type='primary')

sample = st.session_state.get('random_sample')
if sample and message == sample['message']:
    original_label = 'Spam / Lừa đảo' if sample['label'] == 1 else 'Hợp lệ (Ham)'
    st.info(f'Nhãn gốc trong tập test: {original_label}')

if submitted:
    try:
        label, cleaned = predict_message(message, model, vectorizer)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        if label == 1:
            st.error('Dự đoán: SPAM / LỪA ĐẢO')
        else:
            st.success('Dự đoán: HỢP LỆ (HAM)')
        with st.expander('Xem nội dung sau tiền xử lý'):
            st.text(cleaned)
        comparison_path = ARTIFACT_DIR / 'demo_comparison.joblib'
        try:
            bundle = cached_comparison(str(comparison_path), comparison_path.stat().st_mtime_ns)
            rows, neighbors = compare_message(message, bundle)
        except (OSError, ValueError, KeyError):
            st.info('Chạy lại scripts/export_demo_model.py để chuẩn bị đủ bốn cấu hình và dữ liệu giải thích.')
        else:
            st.subheader('So sánh bốn cấu hình')
            st.table(rows)
            st.subheader('Vì sao KNN đưa ra dự đoán này?')
            st.caption('Mỗi cấu hình bỏ phiếu theo 3 tin gần nhất trong tập train. Khoảng cách cosine càng nhỏ, vector càng gần nhau; đây không phải xác suất tin nhắn an toàn.')
            for name, items in neighbors.items():
                spam_votes = sum(item['label'] for item in items)
                with st.expander(f'{name} · {spam_votes}/3 phiếu Spam', expanded=name.endswith('Count')):
                    for rank, item in enumerate(items, 1):
                        label_name = 'Spam' if item['label'] else 'Ham'
                        st.markdown(f"**Láng giềng {rank} · {label_name} · khoảng cách {item['distance']:.4f}**")
                        st.text(item['message'])
                        st.caption(f"Dòng train {item['row']} (đếm từ 0)")

st.caption('Kết quả là dự đoán từ nội dung, không xác minh danh tính người gửi hay độ an toàn của liên kết.')
