import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import time

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="Iris SVM Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# ĐƯỜNG DẪN TÀI NGUYÊN
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_iris.pkl"
HERO_PATH = BASE_DIR / "hero.png.jpg"

# =========================================================
# CSS TÙY CHỈNH CHUYÊN NGHIỆP
# =========================================================
st.markdown("""
<style>
/* Nền toàn trang */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(255, 182, 213, 0.3), transparent 40%),
                radial-gradient(circle at 90% 20%, rgba(255, 214, 231, 0.4), transparent 40%),
                linear-gradient(135deg, #fff5fa 0%, #ffeaf3 50%, #fff8fb 100%);
}

/* Ẩn các thành phần mặc định của Streamlit */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Card chính */
.hero-card, .input-card, .result-card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.9);
    box-shadow: 0 10px 35px rgba(194,24,91,0.08);
    margin-bottom: 25px;
}

/* Tiêu đề */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(90deg, #c2185b, #e91e63, #ad1457);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #87506c;
    font-size: 18px;
    margin-bottom: 30px;
    font-weight: 500;
}

.section-title {
    color: #c2185b;
    font-size: 24px;
    font-weight: 700;
    margin: 20px 0 15px 0;
}

/* Các box tính năng */
.feature {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #ffd1e3;
    box-shadow: 0 4px 15px rgba(194,24,91,0.05);
    transition: transform 0.2s;
}
.feature:hover {
    transform: translateY(-5px);
}
.feature-icon { font-size: 32px; }
.feature-title { font-weight: 700; color: #c2185b; margin-top: 10px; }
.feature-text { font-size: 14px; color: #876074; margin-top: 5px; }

/* Nút bấm (Button) */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #e91e63, #c2185b);
    box-shadow: 0 8px 20px rgba(194,24,91,0.25);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(194,24,91,0.35);
    background: linear-gradient(135deg, #ec407a, #ad1457);
}

/* Badge (Nhãn) */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    background: #ffe0ed;
    color: #c2185b;
    font-weight: 700;
    font-size: 14px;
    margin-top: 10px;
}

/* Footer */
.custom-footer {
    text-align: center;
    color: #96657e;
    font-size: 14px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(194,24,91,0.15);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TẢI MÔ HÌNH
# =========================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return None

model = load_model()

# =========================================================
# SIDEBAR (BẢNG ĐIỀU KHIỂN BÊN)
# =========================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", use_container_width=True)
    st.markdown("### 🌸 Về dự án này")
    st.info("Ứng dụng phân loại hoa Iris tự động sử dụng thuật toán Support Vector Machine (SVM) được huấn luyện trên tập dữ liệu Iris kinh điển của Fisher.")
    st.markdown("---")
    st.markdown("### 👨‍💻 Thông tin")
    st.markdown("- **Môn học:** Đồ án DA1\n- **Công nghệ:** Python, Scikit-learn, Streamlit")

# =========================================================
# GIAO DIỆN CHÍNH
# =========================================================
st.markdown('<div class="main-title">🌸 Iris SVM Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Trí tuệ nhân tạo nhận diện loài hoa Iris qua kích thước đài và cánh hoa</div>', unsafe_allow_html=True)

if model is None:
    st.error("❌ Không thể tải mô hình. Vui lòng kiểm tra lại file `svm_iris.pkl` có nằm cùng thư mục với script này không.")
    st.stop()

# =========================================================
# TABS (ĐIỀU HƯỚNG WEBSITE)
# =========================================================
tab1, tab2 = st.tabs(["🔮 Bảng Dự Đoán", "📖 Hướng Dẫn Kỹ Thuật"])

with tab1:
    # HERO SECTION
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        if HERO_PATH.exists():
            st.image(str(HERO_PATH), use_container_width=True)
        else:
            st.info("💡 Mẹo: Đặt file `hero.png.jpg` vào cùng thư mục để hiển thị ảnh bìa.")
            
    with col2:
        st.markdown("""
        <h3 style="color:#c2185b; margin-top:0;">🤖 Hệ thống Nhận diện Thông minh</h3>
        <p style="color:#70465c; font-size:16px; line-height:1.6;">
        Hãy điều chỉnh 4 thanh trượt bên dưới tương ứng với kích thước thực tế của bông hoa bạn đang có. 
        Mô hình Machine Learning sẽ ngay lập tức tính toán và đưa ra kết quả phân loại với độ chính xác cao nhất.
        </p>
        <div class="badge">Độ chính xác mô hình: ~96%</div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # INPUT SECTION
    st.markdown('<div class="section-title">📋 Thông số đầu vào (cm)</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("🌱 Chiều dài đài hoa (Sepal Length)", 4.0, 8.0, 5.1, 0.1)
        petal_length = st.slider("🌸 Chiều dài cánh hoa (Petal Length)", 1.0, 7.0, 1.4, 0.1)
    with c2:
        sepal_width = st.slider("🍃 Chiều rộng đài hoa (Sepal Width)", 2.0, 4.5, 3.5, 0.1)
        petal_width = st.slider("🌺 Chiều rộng cánh hoa (Petal Width)", 0.1, 2.5, 0.2, 0.1)
        
    st.markdown('</div>', unsafe_allow_html=True)

    # NÚT DỰ ĐOÁN
    if st.button("🔍 PHÂN TÍCH VÀ DỰ ĐOÁN"):
        with st.spinner("Đang chạy thuật toán SVM..."):
            time.sleep(0.8) # Tạo hiệu ứng UX mượt mà
            
            # Xử lý dữ liệu
            data = pd.DataFrame(
                [[sepal_length, sepal_width, petal_length, petal_width]],
                columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
            )
            
            prediction = str(model.predict(data)[0])
            
            flower_info = {
                "Iris-setosa": {
                    "name": "Iris Setosa", "emoji": "🌼", "image": "setosa.jpg",
                    "desc": "Đặc trưng: Cánh hoa nhỏ, ngắn và rộng. Thường mọc ở các vùng khí hậu ôn đới."
                },
                "Iris-versicolor": {
                    "name": "Iris Versicolor", "emoji": "🌷", "image": "versicolor.jpg",
                    "desc": "Đặc trưng: Kích thước trung bình, màu sắc pha trộn. Phân bố nhiều ở Bắc Mỹ."
                },
                "Iris-virginica": {
                    "name": "Iris Virginica", "emoji": "🌺", "image": "virginica.jpg",
                    "desc": "Đặc trưng: Cánh hoa lớn, thuôn dài và màu sắc sặc sỡ. Thích môi trường đất ngập nước."
                }
            }
            
            flower = flower_info.get(prediction, {
                "name": prediction, "emoji": "🌸", "image": None,
                "desc": "Đã hoàn tất phân loại dựa trên trọng số mô hình."
            })

            # HIỂN THỊ KẾT QUẢ
            st.markdown(f"""
            <div class="result-card" style="text-align: center;">
                <div style="color: #8a5570; font-weight: 600; text-transform: uppercase;">Kết quả nhận diện</div>
                <div style="font-size: 70px; margin: 10px 0;">{flower["emoji"]}</div>
                <h1 style="color: #c2185b; margin: 0;">{flower["name"]}</h1>
                <p style="color: #77566a; font-size: 16px; margin-top: 15px;">{flower["desc"]}</p>
                <div class="badge">Hoàn thành tính toán SVM</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Hiển thị ảnh nếu có
            if flower["image"] and (BASE_DIR / flower["image"]).exists():
                st.image(str(BASE_DIR / flower["image"]), use_container_width=True, caption=flower["name"])

            # BIỂU ĐỒ XÁC SUẤT (PROBABILITY)
            if hasattr(model, "predict_proba"):
                st.markdown('<div class="section-title">📊 Biểu đồ tin cậy (Confidence)</div>', unsafe_allow_html=True)
                probabilities = model.predict_proba(data)[0]
                classes = model.classes_
                
                for cls, prob in zip(classes, probabilities):
                    pct = prob * 100
                    st.markdown(f"**{cls}** ({pct:.1f}%)")
                    st.progress(int(pct))

with tab2:
    st.markdown("### Đặc trưng phân loại")
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🌱", "Sepal Length", "Đoạn gốc - Đài hoa"),
        ("🍃", "Sepal Width", "Độ rộng - Đài hoa"),
        ("🌸", "Petal Length", "Chiều dài - Cánh hoa"),
        ("🌺", "Petal Width", "Độ hẹp - Cánh hoa"),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
            <div class="feature">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-text">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("### Khối lượng kỹ thuật")
    st.write("- **Thuật toán:** SVC (Support Vector Classifier) thuộc thư viện Scikit-learn.")
    st.write("- **Hàm kernel:** Tối ưu hóa phân chia ranh giới nhiều lớp (Multi-class classification).")
    st.write("- **Tập dữ liệu:** 150 mẫu hoa chia đều cho 3 lớp (Setosa, Versicolor, Virginica).")

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="custom-footer">
    🌸 <b>Iris SVM Classification WebApp</b><br>
    Đồ án DA1 • Xây dựng bằng Python & Streamlit<br>
    <i>Tối ưu hóa giao diện năm 2024</i>
</div>
""", unsafe_allow_html=True)
