import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Iris SVM Classifier",
    page_icon="🌸",
    layout="centered"
)

# =========================
# ĐƯỜNG DẪN THƯ MỤC
# =========================
BASE_DIR = Path(__file__).resolve().parent


# =========================
# CSS GIAO DIỆN
# =========================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #ffd6e7,
        #ffeaf3,
        #ffc1dc
    );
}

/* =========================
   HOA TRANG TRÍ HAI BÊN
   ========================= */

.flower-left,
.flower-right {
    position: fixed;
    top: 120px;
    z-index: 999;
    font-size: 55px;
    line-height: 1.7;
    text-align: center;
    pointer-events: none;
}

.flower-left {
    left: 15px;
}

.flower-right {
    right: 15px;
}


/* Tiêu đề */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
    color: #c2185b;
}

/* Phụ đề */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #8e4164;
    margin-bottom: 25px;
}

/* Ảnh hero */
.hero-box {
    text-align: center;
    margin-bottom: 25px;
}

/* Card */
.card {
    padding: 25px;
    border-radius: 15px;
    background-color: rgba(255, 255, 255, 0.88);
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(194, 24, 91, 0.15);
}

/* Kết quả */
.result-box {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background-color: rgba(255, 255, 255, 0.92);
    margin-top: 25px;
    box-shadow: 0 5px 20px rgba(194, 24, 91, 0.20);
}

/* Tên loài */
.result-name {
    font-size: 30px;
    font-weight: 700;
    color: #c2185b;
    margin-top: 15px;
}

/* Footer */
.info {
    text-align: center;
    color: #8e4164;
    font-size: 14px;
    margin-top: 30px;
}

/* Trên màn hình nhỏ thì ẩn hoa */
@media (max-width: 900px) {
    .flower-left,
    .flower-right {
        display: none;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================
# HOA TRANG TRÍ HAI BÊN
# =========================
st.markdown("""
<div class="flower-left">
🌸<br>
🌷<br>
🌺<br>
🌼
</div>

<div class="flower-right">
🌺<br>
🌼<br>
🌷<br>
🌸
</div>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
model = joblib.load(BASE_DIR / "svm_iris.pkl")


# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">🌸 Iris SVM Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ứng dụng phân loại hoa Iris sử dụng mô hình học máy SVM'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# ẢNH HERO
# =========================
st.markdown(
    '<div class="hero-box">',
    unsafe_allow_html=True
)

st.image(
    str(BASE_DIR / "hero.png.jpg"),
    width=350
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================
# GIỚI THIỆU
# =========================
st.markdown("""
<div class="card">

### 🤖 Mô hình SVM

Hệ thống sử dụng thuật toán **Support Vector Machine (SVM)**
để phân loại hoa Iris dựa trên 4 đặc trưng:

- 🌱 Sepal Length
- 🌱 Sepal Width
- 🌸 Petal Length
- 🌸 Petal Width

</div>
""", unsafe_allow_html=True)


# =========================
# NHẬP DỮ LIỆU
# =========================
st.subheader("📋 Nhập thông số hoa Iris")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


# =========================
# NÚT DỰ ĐOÁN
# =========================
st.markdown("")

if st.button(
    "🔍 DỰ ĐOÁN LOÀI HOA",
    use_container_width=True
):

    data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    )


    # =========================
    # DỰ ĐOÁN
    # =========================
    prediction = model.predict(data)[0]


    # =========================
    # THÔNG TIN TỪNG LOÀI
    # =========================
    flower_info = {

        "Iris-setosa": {
            "name": "Iris Setosa",
            "emoji": "🌼",
            "image": "setosa.jpg"
        },

        "Iris-versicolor": {
            "name": "Iris Versicolor",
            "emoji": "🌷",
            "image": "versicolor.jpg"
        },

        "Iris-virginica": {
            "name": "Iris Virginica",
            "emoji": "🌺",
            "image": "virginica.jpg"
        }
    }


    # =========================
    # LẤY THÔNG TIN KẾT QUẢ
    # =========================
    flower = flower_info.get(
        prediction,
        {
            "name": prediction,
            "emoji": "🌸",
            "image": "hero.png.jpg"
        }
    )


    # =========================
    # HIỂN THỊ KẾT QUẢ
    # =========================
    st.markdown(
        """
        <div class="result-box">
        <div style="font-size: 24px;">
        🌸 KẾT QUẢ DỰ ĐOÁN
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================
    # HÌNH ẢNH + TÊN
    # =========================
    col_img, col_text = st.columns([1, 1])

    with col_img:

        image_path = BASE_DIR / flower["image"]

        if image_path.exists():

            st.image(
                str(image_path),
                width=250
            )

        else:

            st.warning(
                f"Không tìm thấy ảnh: {flower['image']}"
            )


    with col_text:

        st.markdown(
            f"""
            <div style="
                text-align:center;
                padding-top:60px;
            ">

            <div style="font-size:55px;">
            {flower["emoji"]}
            </div>

            <div class="result-name">
            {flower["name"]}
            </div>

            <p style="
                font-size:17px;
                color:#777;
            ">
            Mô hình SVM dự đoán
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="info">
    SVM Iris Classification • Đồ án DA1
    • Python + Scikit-learn + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)


