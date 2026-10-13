from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


# ไฟล์ทั้งหมดอ้างอิงจากตำแหน่งของ app.py จึงรันจากโฟลเดอร์ไหนก็ได้
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "C02_loan_approval_classification.xlsx - Raw_Data.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_FILE = ARTIFACTS / "best_model.joblib"
COMPARISON_FILE = ARTIFACTS / "model_comparison.csv"
METADATA_FILE = ARTIFACTS / "best_model_metadata.json"

FEATURES = [
    "income_monthly",
    "debt_monthly",
    "credit_score",
    "employment_years",
    "loan_amount",
    "home_status",
]
NUMERIC_FEATURES = FEATURES[:-1]
HOME_STATUS_OPTIONS = ["Family", "Own", "Rent"]

st.set_page_config(
    page_title="ระบบทำนายการอนุมัติสินเชื่อ",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; }
    [data-testid="stSidebar"] { background: #102a43; }
    [data-testid="stSidebar"] * { color: white; }
    .title { color: #102a43; font-size: 2.2rem; font-weight: 700; }
    .hint { color: #52606d; font-size: 1rem; }
    .status-box {
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 1rem;
        background: #f8fbff;
        margin-bottom: 0.75rem;
    }
    .status-label { color: #334e68; font-weight: 600; }
    .status-number { color: #102a43; font-size: 1.6rem; font-weight: 700; }
    .progress-track {
        background: #e6edf5;
        border-radius: 10px;
        height: 12px;
        margin-top: 0.5rem;
    }
    .progress-fill {
        background: #2f80ed;
        border-radius: 10px;
        height: 12px;
    }
    .column-card {
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        background: #ffffff;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_file(path: Path, label: str) -> None:
    if not path.exists():
        st.error(f"ไม่พบไฟล์ {label}: {path.relative_to(ROOT)}")
        st.info("กรุณาตรวจสอบว่าเปิดเว็บจากโฟลเดอร์โปรเจกต์ที่มีไฟล์ artifacts แล้ว")
        st.stop()


@st.cache_data
def read_data() -> pd.DataFrame:
    check_file(DATA_FILE, "Dataset")
    return pd.read_csv(DATA_FILE)


@st.cache_data
def read_comparison() -> pd.DataFrame:
    check_file(COMPARISON_FILE, "model_comparison.csv")
    return pd.read_csv(COMPARISON_FILE)


@st.cache_data
def read_metadata() -> dict:
    check_file(METADATA_FILE, "best_model_metadata.json")
    with METADATA_FILE.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def read_model():
    check_file(MODEL_FILE, "best_model.joblib")
    return joblib.load(MODEL_FILE)


def heading(title: str, explanation: str) -> None:
    st.markdown(f'<div class="title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hint">{explanation}</div>', unsafe_allow_html=True)
    st.divider()


def dashboard() -> None:
    heading(
        "Loan Approval Prediction System",
        "ระบบตัวอย่างสำหรับดูข้อมูล เปรียบเทียบโมเดล และทำนายผลการอนุมัติสินเชื่อ",
    )

    data = read_data()
    approved = int((data["approved"] == 1).sum())
    not_approved = int((data["approved"] == 0).sum())

    a, b, c = st.columns(3)
    a.metric("จำนวนข้อมูลทั้งหมด", f"{len(data):,} รายการ")
    b.metric("อนุมัติ", f"{approved:,} รายการ")
    c.metric("ไม่อนุมัติ", f"{not_approved:,} รายการ")

    st.subheader("ข้อมูลใน Dataset")
    st.write(
        f"Dataset มีทั้งหมด **{data.shape[0]:,} แถว** และ **{data.shape[1]:,} คอลัมน์** "
        "โดยใช้คอลัมน์ `approved` เป็นผลลัพธ์ที่ต้องการทำนาย"
    )

    st.subheader("สัดส่วนผลลัพธ์")
    total = approved + not_approved
    approved_percent = approved / total * 100
    not_approved_percent = not_approved / total * 100

    left, right = st.columns(2)
    for container, label, count, percentage in [
        (left, "อนุมัติ", approved, approved_percent),
        (right, "ไม่อนุมัติ", not_approved, not_approved_percent),
    ]:
        with container:
            st.markdown(
                f"""
                <div class="status-box">
                    <div class="status-label">{label}</div>
                    <div class="status-number">{count:,} รายการ ({percentage:.2f}%)</div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: {percentage:.2f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("8 คอลัมน์ที่มีใน Dataset")
    st.caption("รายชื่อคอลัมน์ดึงจากไฟล์ Dataset จริง")
    column_left, column_right = st.columns(2)
    midpoint = (len(data.columns) + 1) // 2
    for container, columns in [
        (column_left, data.columns[:midpoint]),
        (column_right, data.columns[midpoint:]),
    ]:
        with container:
            for column in columns:
                number = data.columns.get_loc(column) + 1
                st.markdown(
                    f'<div class="column-card"><b>{number}.</b> {column}</div>',
                    unsafe_allow_html=True,
                )


def model_comparison() -> None:
    heading(
        "เปรียบเทียบโมเดล",
        "ผลลัพธ์ดึงจากไฟล์ model_comparison.csv ที่บันทึกจากการฝึกโมเดลจริง",
    )

    comparison = read_comparison()
    metadata = read_metadata()
    required = {
        "Algorithm",
        "Best CV F1",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
    }
    missing = required.difference(comparison.columns)
    if missing:
        st.error(f"CSV ขาดคอลัมน์: {', '.join(sorted(missing))}")
        st.stop()

    best_name = metadata["best_algorithm"]
    best_cv_f1 = float(metadata["best_cv_f1"])
    st.success(f"โมเดลที่ดีที่สุด: {best_name} | Best CV F1 = {best_cv_f1:.4f}")
    st.caption("จัดอันดับจาก Mean CV F1 บน Training Set ไม่ได้เลือกใหม่จาก Test Set")

    columns = [
        "Algorithm",
        "Best CV F1",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
    ]
    st.dataframe(
        comparison[columns].style.format({column: "{:.4f}" for column in columns[1:]}),
        width="stretch",
        hide_index=True,
    )

    st.subheader("กราฟเปรียบเทียบผลการทดสอบ")
    st.bar_chart(
        comparison.set_index("Algorithm")[
            ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
        ],
        color=["#2f80ed", "#56ccf2", "#102a43", "#27ae60"],
    )


def prediction() -> None:
    heading(
        "ทำนายผลการอนุมัติสินเชื่อ",
        "กรอกข้อมูลผู้สมัคร แล้วกดปุ่มเพื่อใช้โมเดลที่บันทึกไว้ทำนายผล",
    )

    data = read_data()
    missing = set(FEATURES).difference(data.columns)
    if missing:
        st.error(f"Dataset ขาด Feature: {', '.join(sorted(missing))}")
        st.stop()

    model = read_model()
    home_values = HOME_STATUS_OPTIONS

    st.info("กรอกข้อมูลให้ครบทุกช่อง ตัวเลขต้องเป็นค่าศูนย์หรือค่าบวก")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            income = st.number_input("รายได้ต่อเดือน (income_monthly)", min_value=0.0, value=30000.0, step=1000.0)
            debt = st.number_input("หนี้ต่อเดือน (debt_monthly)", min_value=0.0, value=5000.0, step=500.0)
            credit = st.number_input("คะแนนเครดิต (credit_score)", min_value=0.0, max_value=1000.0, value=650.0, step=1.0)
        with right:
            employment = st.number_input("อายุงาน (employment_years)", min_value=0.0, value=3.0, step=1.0)
            loan = st.number_input("จำนวนเงินกู้ (loan_amount)", min_value=0.0, value=100000.0, step=1000.0)
            home = st.selectbox("สถานะที่อยู่อาศัย (home_status)", home_values)

        submitted = st.form_submit_button("เริ่มทำนายผล", type="primary", width="stretch")

    if not submitted:
        return

    input_data = pd.DataFrame(
        [{
            "income_monthly": float(income),
            "debt_monthly": float(debt),
            "credit_score": float(credit),
            "employment_years": float(employment),
            "loan_amount": float(loan),
            "home_status": home,
        }],
        columns=FEATURES,
    )

    if input_data[NUMERIC_FEATURES].isnull().any().any():
        st.error("ข้อมูลตัวเลขไม่ครบหรือไม่ถูกต้อง")
        return

    try:
        result = int(model.predict(input_data)[0])
    except (TypeError, ValueError, KeyError) as error:
        st.error(f"ไม่สามารถทำนายได้: {error}")
        return

    st.subheader("ผลการทำนาย")
    if result == 1:
        st.success("ผลการทำนาย: Approved (มีแนวโน้มได้รับอนุมัติ)")
    elif result == 0:
        st.error("ผลการทำนาย: Not Approved (มีแนวโน้มไม่ได้รับอนุมัติ)")
    else:
        st.error(f"ค่าที่โมเดลส่งกลับมาไม่ถูกต้อง: {result}")
        return

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = dict(zip(model.classes_, probabilities))
        st.info(
            f"ความน่าจะเป็นของผลที่ทำนาย: {classes[result]:.2%} "
            "(เป็นค่าจากโมเดล ไม่ใช่การรับประกันผลอนุมัติ)"
        )
    else:
        st.info("โมเดลนี้ไม่มี predict_proba จึงแสดงเฉพาะผลการทำนาย")


with st.sidebar:
    st.title("🏦 Loan Approval")
    st.caption("เมนูหลัก")
    page = st.radio(
        "เลือกหน้าที่ต้องการ",
        ["หน้าแรก", "เปรียบเทียบโมเดล", "ทำนายผลสินเชื่อ"],
    )
    st.divider()
    st.caption("เปิดด้วยคำสั่ง: streamlit run app.py")


if page == "หน้าแรก":
    dashboard()
elif page == "เปรียบเทียบโมเดล":
    model_comparison()
else:
    prediction()
