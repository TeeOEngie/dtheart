import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับความสวยงาม
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 20px;
        padding: 10px 30px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .prediction-result {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_disease_model.pkl')
        return model
    except:
        st.error("ไม่พบไฟล์โมเดล กรุณาอัปโหลดไฟล์ heart_disease_model.pkl")
        return None

model = load_model()

# Header
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1 style='color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;'>
        ❤️ Heart Disease Prediction System
    </h1>
    <p style='font-size: 1.2rem; color: #666;'>
        ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar สำหรับข้อมูลอินพุต
st.sidebar.markdown("## 📋 ข้อมูลผู้ป่วย")

# สร้างฟอร์มสำหรับกรอกข้อมูล
with st.sidebar.form("prediction_form"):
    st.markdown("### 🎯 ข้อมูลพื้นฐาน")
    age = st.slider("อายุ (ปี)", 20, 100, 50)
    sex = st.selectbox("เพศ", ["ชาย", "หญิง"])
    sex_value = 1 if sex == "ชาย" else 0
    
    st.markdown("### 💓 อาการและสัญญาณชีพ")
    chest_pain_type = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก",
        ["แบบที่ 1 (Typical Angina)", "แบบที่ 2 (Atypical Angina)", 
         "แบบที่ 3 (Non-anginal Pain)", "แบบที่ 4 (Asymptomatic)"]
    )
    chest_pain_map = {"แบบที่ 1 (Typical Angina)": 1, "แบบที่ 2 (Atypical Angina)": 2,
                      "แบบที่ 3 (Non-anginal Pain)": 3, "แบบที่ 4 (Asymptomatic)": 4}
    chest_pain_value = chest_pain_map[chest_pain_type]
    
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("ระดับคอเลสเตอรอล (mg/dl)", 100, 600, 200)
    
    st.markdown("### 🔬 ผลการตรวจ")
    fasting_bs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl", ["ไม่ใช่", "ใช่"])
    fasting_bs_value = 1 if fasting_bs == "ใช่" else 0
    
    resting_ecg = st.selectbox(
        "ผล ECG ขณะพัก",
        ["ปกติ (0)", "มีความผิดปกติ (1)", "แสดงภาวะ LV hypertrophy (2)"]
    )
    ecg_map = {"ปกติ (0)": 0, "มีความผิดปกติ (1)": 1, "แสดงภาวะ LV hypertrophy (2)": 2}
    resting_ecg_value = ecg_map[resting_ecg]
    
    max_hr = st.number_input("อัตราการเต้นของหัวใจสูงสุด (bpm)", 60, 220, 150)
    
    exercise_angina = st.selectbox("มีอาการเจ็บหน้าอกขณะออกกำลังกาย", ["ไม่มี", "มี"])
    exercise_angina_value = 1 if exercise_angina == "มี" else 0
    
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
    
    st_slope = st.selectbox(
        "ST Slope",
        ["Up sloping (1)", "Flat (2)", "Down sloping (3)"]
    )
    slope_map = {"Up sloping (1)": 1, "Flat (2)": 2, "Down sloping (3)": 3}
    st_slope_value = slope_map[st_slope]
    
    submitted = st.form_submit_button("🔍 ทำนายผล", use_container_width=True)

# Main content
if submitted and model is not None:
    # สร้าง DataFrame สำหรับทำนาย
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_value],
        'ChestPainType': [chest_pain_value],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs_value],
        'RestingECG': [resting_ecg_value],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina_value],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope_value]
    })
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    
    # แสดงผลลัพธ์
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if prediction == 1:
            st.markdown("""
            <div class='prediction-result high-risk'>
                ⚠️ ผลทำนาย: มีความเสี่ยงเป็นโรคหัวใจ<br>
                <span style='font-size: 18px;'>ควรปรึกษาแพทย์เพื่อตรวจเพิ่มเติม</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='prediction-result low-risk'>
                ✅ ผลทำนาย: ไม่พบความเสี่ยงเป็นโรคหัวใจ<br>
                <span style='font-size: 18px;'>รักษาสุขภาพต่อไปนะครับ</span>
            </div>
            """, unsafe_allow_html=True)
    
    # แสดงข้อมูลที่ใช้ทำนาย
    st.markdown("### 📊 ข้อมูลที่ใช้ในการทำนาย")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("อายุ", f"{age} ปี")
        st.metric("เพศ", sex)
    
    with col2:
        st.metric("ความดันโลหิต", f"{resting_bp} mm Hg")
        st.metric("คอเลสเตอรอล", f"{cholesterol} mg/dl")
    
    with col3:
        st.metric("อัตราการเต้นหัวใจสูงสุด", f"{max_hr} bpm")
        st.metric("Oldpeak", f"{oldpeak}")
    
    with col4:
        st.metric("อาการเจ็บหน้าอก", chest_pain_type.split()[0])
        st.metric("ST Slope", st_slope.split()[0])
    
    # คำแนะนำ
    st.markdown("### 💡 คำแนะนำ")
    if prediction == 1:
        st.warning("""
        **จากผลการวิเคราะห์ คุณมีความเสี่ยงที่จะเป็นโรคหัวใจ**
        
        แนะนำให้:
        - ปรึกษาแพทย์ผู้เชี่ยวชาญโดยเร็ว
        - ตรวจสุขภาพอย่างละเอียด
        - ปรับเปลี่ยนพฤติกรรมสุขภาพ
        - ออกกำลังกายสม่ำเสมอ
        - ควบคุมอาหารและน้ำหนัก
        """)
    else:
        st.success("""
        **จากผลการวิเคราะห์ คุณไม่มีความเสี่ยงที่จะเป็นโรคหัวใจ**
        
        แต่ควรดูแลสุขภาพต่อไปโดย:
        - ออกกำลังกายสม่ำเสมอ
        - ทานอาหารที่มีประโยชน์
        - พักผ่อนให้เพียงพอ
        - ตรวจสุขภาพประจำปี
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #666;'>
    <p>สร้างด้วย ❤️ โดย Machine Learning | Decision Tree Model</p>
    <p style='font-size: 12px;'>⚠️ คำเตือน: ผลการทำนายเป็นเพียงการประเมินเบื้องต้น ไม่สามารถทดแทนการวินิจฉัยของแพทย์ได้</p>
</div>
""", unsafe_allow_html=True)