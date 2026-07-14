import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================
# ตั้งค่าหน้าเว็บ
# ============================
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS - ดีไซน์สวยงาม
# ============================
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #e63946 0%, #f77f7f 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(230, 57, 70, 0.3);
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] h2 {
        color: #e63946;
        font-weight: 700;
    }
    
    /* ปุ่มทำนาย */
    .stButton>button {
        background: linear-gradient(135deg, #e63946 0%, #f77f7f 100%);
        color: white !important;
        border-radius: 30px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 6px 15px rgba(230, 57, 70, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(230, 57, 70, 0.5);
    }
    
    /* กล่องผลลัพธ์ */
    .result-box {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-high {
        background: linear-gradient(135deg, #e63946 0%, #ff6b6b 100%);
        color: white;
    }
    .result-low {
        background: linear-gradient(135deg, #06a77d 0%, #51cf66 100%);
        color: white;
    }
    .result-box h2 {
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
        font-weight: 800;
    }
    .result-box p {
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 5px solid #e63946;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.5rem;
        color: #e63946;
        font-weight: 800;
    }
    
    /* Info Box */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid #4dabf7;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# โหลดโมเดล
# ============================
@st.cache_resource
def load_model():
    model_path = Path("heart_disease_model.pkl")
    if not model_path.exists():
        return None
    return joblib.load(model_path)

model = load_model()

# ============================
# Header
# ============================
st.markdown("""
<div class='main-header'>
    <h1>❤️ Heart Disease Predictor</h1>
    <p>ระบบทำนายความเสี่ยงโรคหัวใจด้วย Machine Learning (Decision Tree)</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Sidebar - ฟอร์มกรอกข้อมูล
# ============================
with st.sidebar:
    st.markdown("## 📋 ข้อมูลผู้ป่วย")
    st.markdown("---")
    
    with st.form("prediction_form"):
        # ข้อมูลพื้นฐาน
        st.markdown("### 👤 ข้อมูลพื้นฐาน")
        age = st.slider("🎂 อายุ (ปี)", 20, 100, 50)
        sex = st.selectbox("⚧️ เพศ", ["ชาย", "หญิง"])
        sex_value = 1 if sex == "ชาย" else 0
        
        st.markdown("---")
        st.markdown("### 💓 อาการและสัญญาณชีพ")
        
        # ChestPainType: 1=ATA, 2=NAP, 3=ASY, 4=TA (ตาม Heart4.csv)
        chest_pain_type = st.selectbox(
            "🫁 ประเภทอาการเจ็บหน้าอก",
            ["ASY - ไม่มีอาการ (3)",
             "NAP - เจ็บหน้าอกไม่เฉพาะ (2)",
             "ATA - เจ็บหน้าอกรูปแบบเฉพาะ (1)",
             "TA - เจ็บหน้าอกรูปแบบทั่วไป (4)"]
        )
        chest_map = {
            "ASY - ไม่มีอาการ (3)": 3,
            "NAP - เจ็บหน้าอกไม่เฉพาะ (2)": 2,
            "ATA - เจ็บหน้าอกรูปแบบเฉพาะ (1)": 1,
            "TA - เจ็บหน้าอกรูปแบบทั่วไป (4)": 4
        }
        chest_pain_value = chest_map[chest_pain_type]
        
        resting_bp = st.number_input(
            "💉 ความดันโลหิตขณะพัก (mm Hg)", 
            min_value=80, max_value=200, value=120, step=1
        )
        cholesterol = st.number_input(
            "🩸 ระดับคอเลสเตอรอล (mg/dl)", 
            min_value=0, max_value=600, value=200, step=1
        )
        
        st.markdown("---")
        st.markdown("### 🔬 ผลการตรวจ")
        
        fasting_bs = st.selectbox(
            "🍬 น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl",
            ["ไม่ใช่ (0)", "ใช่ (1)"]
        )
        fasting_bs_value = 1 if "ใช่" in fasting_bs else 0
        
        # RestingECG: 1=normal, 2=ST-T abnormality, 3=LV hypertrophy
        resting_ecg = st.selectbox(
            "📈 ผล ECG ขณะพัก",
            ["Normal - ปกติ (1)",
             "ST-T wave abnormality (2)",
             "LV hypertrophy (3)"]
        )
        ecg_map = {
            "Normal - ปกติ (1)": 1,
            "ST-T wave abnormality (2)": 2,
            "LV hypertrophy (3)": 3
        }
        resting_ecg_value = ecg_map[resting_ecg]
        
        max_hr = st.number_input(
            "💓 อัตราการเต้นหัวใจสูงสุด (bpm)",
            min_value=60, max_value=220, value=150, step=1
        )
        
        exercise_angina = st.selectbox(
            "🏃 อาการเจ็บหน้าอกขณะออกกำลังกาย",
            ["ไม่มี (0)", "มี (1)"]
        )
        exercise_angina_value = 1 if "มี" in exercise_angina else 0
        
        oldpeak = st.number_input(
            "📉 Oldpeak (ST depression)",
            min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
            format="%.1f"
        )
        
        # ST_Slope: 1=upsloping, 2=flat, 3=downsloping
        st_slope = st.selectbox(
            "📊 ST Slope",
            ["Up sloping - ขึ้น (1)",
             "Flat - แบน (2)",
             "Down sloping - ลง (3)"]
        )
        slope_map = {
            "Up sloping - ขึ้น (1)": 1,
            "Flat - แบน (2)": 2,
            "Down sloping - ลง (3)": 3
        }
        st_slope_value = slope_map[st_slope]
        
        st.markdown("---")
        submitted = st.form_submit_button(
            "🔍 ทำนายผล",
            use_container_width=True
        )

# ============================
# Main Content
# ============================
if model is None:
    st.error("❌ ไม่พบไฟล์ `heart_disease_model.pkl`")
    st.info("""
    **วิธีแก้ไข:**
    1. รันโค้ดใน Google Colab เพื่อเทรนโมเดล
    2. ดาวน์โหลดไฟล์ `heart_disease_model.pkl`
    3. วางไฟล์ไว้ในโฟลเดอร์เดียวกับ `app.py`
    """)
    st.stop()

if submitted:
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
    
    # ============================
    # แสดงผลลัพธ์
    # ============================
    if prediction == 1:
        st.markdown("""
        <div class='result-box result-high'>
            <h2>⚠️ พบความเสี่ยงโรคหัวใจ</h2>
            <p>ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติมโดยด่วน</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-box result-low'>
            <h2>✅ ไม่พบความเสี่ยงโรคหัวใจ</h2>
            <p>ผลการวิเคราะห์เบื้องต้นอยู่ในเกณฑ์ปกติ</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================
    # แสดงข้อมูลที่ใช้ทำนาย (Metric Cards)
    # ============================
    st.markdown("### 📊 ข้อมูลที่ใช้ในการทำนาย")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🎂 อายุ</div>
            <div class='metric-value'>{age} ปี</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>💉 ความดันโลหิต</div>
            <div class='metric-value'>{resting_bp} mmHg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>💓 ชีพจรสูงสุด</div>
            <div class='metric-value'>{max_hr} bpm</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🩸 คอเลสเตอรอล</div>
            <div class='metric-value'>{cholesterol} mg/dl</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>⚧️ เพศ</div>
            <div class='metric-value'>{sex}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🫁 อาการเจ็บหน้าอก</div>
            <div class='metric-value'>{chest_pain_type.split(' - ')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>📈 ECG</div>
            <div class='metric-value'>{resting_ecg.split(' - ')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>📊 ST Slope</div>
            <div class='metric-value'>{st_slope.split(' - ')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ============================
    # Radar Chart - แสดงโปรไฟล์ผู้ป่วย
    # ============================
    st.markdown("### 📈 โปรไฟล์สุขภาพ")
    
    # Normalize ข้อมูลสำหรับ Radar Chart
    features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    values = [age, resting_bp, cholesterol, max_hr, oldpeak]
    max_values = [100, 200, 600, 220, 7]
    
    normalized = [(v / m) * 100 for v, m in zip(values, max_values)]
    normalized.append(normalized[0])  # ปิดวง
    
    categories = features + [features[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=categories,
        fill='toself',
        name='โปรไฟล์ผู้ป่วย',
        line_color='#e63946',
        fillcolor='rgba(230, 57, 70, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================
    # คำแนะนำ
    # ============================
    st.markdown("---")
    st.markdown("### 💡 คำแนะนำ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.markdown("""
            <div class='info-box'>
                <h4 style='color: #e63946; margin-top: 0;'>🚨 การปฏิบัติตัว</h4>
                <ul style='line-height: 1.8;'>
                    <li>✅ ปรึกษาแพทย์ผู้เชี่ยวชาญทันที</li>
                    <li>✅ ตรวจ ECG และ Echocardiogram</li>
                    <li>✅ ตรวจระดับไขมันในเลือด</li>
                    <li>✅ ควบคุมความดันโลหิต</li>
                    <li>✅ หลีกเลี่ยงอาหารเค็ม/มัน</li>
                    <li>✅ งดสูบบุหรี่และแอลกอฮอล์</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <h4 style='color: #06a77d; margin-top: 0;'>✨ การดูแลสุขภาพ</h4>
                <ul style='line-height: 1.8;'>
                    <li>✅ ออกกำลังกายสม่ำเสมอ 150 นาที/สัปดาห์</li>
                    <li>✅ ทานอาหารที่มีประโยชน์</li>
                    <li>✅ นอนหลับ 7-8 ชั่วโมงต่อวัน</li>
                    <li>✅ ตรวจสุขภาพประจำปี</li>
                    <li>✅ จัดการความเครียด</li>
                    <li>✅ ดื่มน้ำสะอาด 8 แก้วต่อวัน</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
            <h4 style='color: #4dabf7; margin-top: 0;'>📚 ข้อมูลเพิ่มเติม</h4>
            <p style='line-height: 1.8;'>
                โรคหัวใจเป็นสาเหตุการเสียชีวิตอันดับต้นๆ ของโลก 
                การป้องกันและการตรวจพบตั้งแต่เนิ่นๆ สามารถลดความเสี่ยงได้มาก
            </p>
            <p style='line-height: 1.8; margin: 0;'>
                <strong>ปัจจัยเสี่ยงหลัก:</strong>
                <br>• ความดันโลหิตสูง
                <br>• คอเลสเตอรอลสูง
                <br>• เบาหวาน
                <br>• การสูบบุหรี่
                <br>• โรคอ้วน
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # หน้าแรก - ยังไม่ทำนาย
    st.markdown("### 🎯 เริ่มต้นใช้งาน")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='info-box' style='border-left-color: #e63946;'>
            <h3 style='color: #e63946; text-align: center;'>1️⃣</h3>
            <h4 style='text-align: center;'>กรอกข้อมูล</h4>
            <p style='text-align: center;'>กรอกข้อมูลสุขภาพในแถบด้านซ้าย</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box' style='border-left-color: #4dabf7;'>
            <h3 style='color: #4dabf7; text-align: center;'>2️⃣</h3>
            <h4 style='text-align: center;'>วิเคราะห์</h4>
            <p style='text-align: center;'>กดปุ่ม "ทำนายผล" เพื่อวิเคราะห์</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='info-box' style='border-left-color: #06a77d;'>
            <h3 style='color: #06a77d; text-align: center;'>3️⃣</h3>
            <h4 style='text-align: center;'>รับผล</h4>
            <p style='text-align: center;'>ดูผลการทำนายและคำแนะนำ</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 เกี่ยวกับโมเดล")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🤖 Decision Tree Classifier**
        
        โมเดลนี้ใช้ алгоритม Decision Tree ซึ่งเป็น Machine Learning ประเภท Classification
        ที่เรียนรู้จากข้อมูลผู้ป่วยจริงเพื่อทำนายความเสี่ยงโรคหัวใจ
        
        **Features ที่ใช้:**
        - Age, Sex, ChestPainType
        - RestingBP, Cholesterol, FastingBS
        - RestingECG, MaxHR, ExerciseAngina
        - Oldpeak, ST_Slope
        """)
    
    with col2:
        st.warning("""
        **⚠️ คำเตือนสำคัญ**
        
        ผลการทำนายนี้เป็นเพียงการประเมินเบื้องต้นจากโมเดล Machine Learning 
        **ไม่สามารถทดแทนการวินิจฉัยของแพทย์ได้**
        
        หากมีอาการผิดปกติ โปรดปรึกษาแพทย์ผู้เชี่ยวชาญทันที
        """)

# ============================
# Footer
# ============================
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>❤️ สร้างด้วย <strong>Streamlit + Machine Learning</strong></p>
    <p style='font-size: 0.9rem; color: #999;'>
        © 2026 Heart Disease Prediction System | Decision Tree Model
    </p>
</div>
""", unsafe_allow_html=True)