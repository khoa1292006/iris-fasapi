import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# CẤU HÌNH
# =========================================================
st.set_page_config(
    page_title="Iris SVM Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# ĐƯỜNG DẪN
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "svm_iris.pkl"
HERO_PATH = BASE_DIR / "hero.png.jpg"

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>

/* =========================
   NỀN TOÀN TRANG
========================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(255, 182, 213, 0.45),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(255, 214, 231, 0.55),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #fff5fa 0%,
            #ffeaf3 45%,
            #fff8fb 100%
        );
}


/* =========================
   ẨN HEADER STREAMLIT
========================= */

header {
    background: transparent !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* =========================
   CONTAINER
========================= */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================
   TITLE
========================= */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #c2185b,
        #e91e63,
        #ad1457
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #87506c;
    font-size: 18px;
    margin-bottom: 35px;
}


/* =========================
   HERO CARD
========================= */

.hero-card {
    background: rgba(255,255,255,0.72);
    backdrop-filter: blur(12px);

    border: 1px solid rgba(255,255,255,0.8);

    border-radius: 28px;

    padding: 25px;

    box-shadow:
        0 15px 45px rgba(194,24,91,0.12);

    margin-bottom: 30px;
}


/* =========================
   SECTION TITLE
========================= */

.section-title {
    color: #c2185b;
    font-size: 27px;
    font-weight: 750;

    margin-top: 15px;
    margin-bottom: 18px;
}


/* =========================
   INFO CARD
========================= */

.info-card {
    background: rgba(255,255,255,0.78);

    border-radius: 22px;

    padding: 25px 30px;

    border: 1px solid rgba(255,255,255,0.9);

    box-shadow:
        0 10px 30px rgba(194,24,91,0.08);

    margin-bottom: 25px;
}

.info-card h3 {
    color: #c2185b;
    margin-bottom: 12px;
}

.info-card p {
    color: #684458;
    font-size: 16px;
    line-height: 1.7;
}


/* =========================
   FEATURE BOX
========================= */

.feature {
    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.95),
        rgba(255,240,247,0.9)
    );

    padding: 18px;

    border-radius: 18px;

    text-align: center;

    border: 1px solid #ffd1e3;

    box-shadow:
        0 6px 20px rgba(194,24,91,0.06);

    margin-bottom: 10px;
}

.feature-icon {
    font-size: 30px;
}

.feature-title {
    font-weight: 700;
    color: #c2185b;
    margin-top: 5px;
}

.feature-text {
    font-size: 14px;
    color: #876074;
}


/* =========================
   INPUT AREA
========================= */

.input-card {
    background: rgba(255,255,255,0.82);

    padding: 28px;

    border-radius: 24px;

    border: 1px solid rgba(255,255,255,0.95);

    box-shadow:
        0 12px 35px rgba(194,24,91,0.10);

    margin-top: 15px;
}


/* =========================
   INPUT LABEL
========================= */

label {
    color: #6b3c55 !important;
    font-weight: 600 !important;
}


/* =========================
   BUTTON
========================= */

.stButton > button {

    width: 100%;

    border-radius: 16px;

    border: none;

    padding: 15px 20px;

    font-size: 18px;

    font-weight: 700;

    color: white;

    background: linear-gradient(
        135deg,
        #e91e63,
        #c2185b
    );

    box-shadow:
        0 8px 20px rgba(194,24,91,0.25);

    transition: all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0 12px 28px rgba(194,24,91,0.35);

    background: linear-gradient(
        135deg,
        #ec407a,
        #ad1457
    );
}


/* =========================
   RESULT
========================= */

.result-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.95),
            rgba(255,237,246,0.95)
        );

    border-radius: 28px;

    padding: 30px;

    margin-top: 30px;

    border: 2px solid #ffd0e2;

    box-shadow:
        0 15px 40px rgba(194,24,91,0.14);

    text-align: center;
}

.result-label {

    font-size: 17px;

    color: #8a5570;

    margin-bottom: 5px;
}

.result-name {

    font-size: 38px;

    font-weight: 800;

    color: #c2185b;

    margin: 8px 0;
}

.result-description {

    color: #77566a;

    font-size: 16px;
}


/* =========================
   BADGE
========================= */

.badge {

    display: inline-block;

    padding: 8px 18px;

    border-radius: 30px;

    background: #ffe0ed;

    color: #c2185b;

    font-weight: 700;

    margin-top: 10px;
}


/* =========================
   FOOTER
========================= */

.footer {

    text-align: center;

    color: #96657e;

    font-size: 14px;

    margin-top: 45px;

    padding-top: 20px;

    border-top: 1px solid rgba(194,24,91,0.12);
}


/* =========================
   MOBILE
========================= */

@media (max-width: 768px) {

    .main-title {
        font-size: 36px;
    }

    .subtitle {
        font-size: 15px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error(f"❌ Không thể tải mô hình: {e}")
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🌸 Iris SVM Classifier
    </div>

    <div class="subtitle">
        Ứng dụng trí tuệ nhân tạo phân loại hoa Iris
        bằng thuật toán Support Vector Machine
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero-card">',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.15, 1])

with col1:

    if HERO_PATH.exists():

        st.image(
            str(HERO_PATH),
            use_container_width=True
        )

with col2:

    st.markdown(
        """
        <div style="padding: 25px 10px;">

        <h2 style="color:#c2185b;">
        🌷 Nhận diện hoa Iris
        </h2>

        <p style="
        color:#70465c;
        font-size:17px;
        line-height:1.8;
        ">

        Hệ thống sử dụng mô hình
        <b>Support Vector Machine (SVM)</b>
        để dự đoán loài hoa Iris dựa trên
        4 đặc trưng hình thái của hoa.

        </p>

        <div class="badge">
        🤖 Machine Learning
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# FEATURES
# =========================================================

st.markdown(
    '<div class="section-title">🌿 4 đặc trưng đầu vào</div>',
    unsafe_allow_html=True
)

f1, f2, f3, f4 = st.columns(4)

features = [
    ("🌱", "Sepal Length", "Chiều dài đài hoa"),
    ("🍃", "Sepal Width", "Chiều rộng đài hoa"),
    ("🌸", "Petal Length", "Chiều dài cánh hoa"),
    ("🌺", "Petal Width", "Chiều rộng cánh hoa"),
]

for col, item in zip(
    [f1, f2, f3, f4],
    features
):

    with col:

        st.markdown(
            f"""
            <div class="feature">

                <div class="feature-icon">
                    {item[0]}
                </div>

                <div class="feature-title">
                    {item[1]}
                </div>

                <div class="feature-text">
                    {item[2]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# INPUT
# =========================================================

st.markdown(
    '<div class="section-title">📋 Nhập thông số hoa</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "🌱 Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "🌸 Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )


with col2:

    sepal_width = st.number_input(
        "🍃 Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "🌺 Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("")

if st.button(
    "🔍  DỰ ĐOÁN LOÀI HOA",
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

    prediction = str(prediction)


    # =========================
    # THÔNG TIN HOA
    # =========================

    flower_info = {

        "Iris-setosa": {
            "name": "Iris Setosa",
            "emoji": "🌼",
            "image": "setosa.jpg",
            "description":
                "Loài Iris có cánh hoa nhỏ và thường có màu tím nhạt."
        },

        "Iris-versicolor": {
            "name": "Iris Versicolor",
            "emoji": "🌷",
            "image": "versicolor.jpg",
            "description":
                "Loài Iris có kích thước trung bình với màu sắc đặc trưng."
        },

        "Iris-virginica": {
            "name": "Iris Virginica",
            "emoji": "🌺",
            "image": "virginica.jpg",
            "description":
                "Loài Iris có cánh hoa lớn hơn và thường có kích thước nổi bật."
        }

    }


    flower = flower_info.get(
        prediction,
        {
            "name": prediction,
            "emoji": "🌸",
            "image": "hero.png.jpg",
            "description":
                "Mô hình đã đưa ra kết quả dự đoán."
        }
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-label">
                🌸 KẾT QUẢ DỰ ĐOÁN
            </div>

            <div style="font-size:65px;">
                {flower["emoji"]}
            </div>

            <div class="result-name">
                {flower["name"]}
            </div>

            <div class="result-description">
                {flower["description"]}
            </div>

            <div class="badge">
                🤖 Dự đoán bởi mô hình SVM
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # IMAGE RESULT
    # =====================================================

    image_path = BASE_DIR / flower["image"]

    if image_path.exists():

        st.markdown(
            "<div style='height:20px'></div>",
            unsafe_allow_html=True
        )

        col_left, col_center, col_right = st.columns(
            [1, 1.4, 1]
        )

        with col_center:

            st.image(
                str(image_path),
                use_container_width=True
            )


    # =====================================================
    # PROBABILITY
    # =====================================================

    if hasattr(model, "predict_proba"):

        try:

            probabilities = model.predict_proba(data)[0]

            classes = model.classes_

            prob_df = pd.DataFrame({
                "Loài hoa": classes,
                "Xác suất": probabilities
            })

            prob_df["Xác suất"] = (
                prob_df["Xác suất"] * 100
            ).round(2)

            st.markdown(
                '<div class="section-title">'
                '📊 Xác suất dự đoán'
                '</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                prob_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception:
            pass


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🌸 <b>Iris SVM Classification</b>

        <br>

        Đồ án DA1 • Python • Scikit-learn • Streamlit

        <br><br>

        Made with ❤️

    </div>
    """,
    unsafe_allow_html=True
)


