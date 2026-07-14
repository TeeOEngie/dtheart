# app.py - รันด้วยคำสั่ง: streamlit run app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# โหลดโมเดล
with open('heart_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🫀 Heart Disease Prediction")
st.write("กรอกข้อมูลเพื่อทำนายความเสี่ยงโรคหัวใจ")

# สร้างฟอร์มรับข้อมูล
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (อายุ)", 20, 100, 50)
    sex = st.selectbox("Sex (เพศ)", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
    resting_bp = st.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", [0, 1, 2])
    max_hr = st.number_input("Max Heart Rate", 50, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", [1, 2, 3])

# ปุ่มทำนาย
if st.button("🔮 Predict"):
    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, 
                            fasting_bs, resting_ecg, max_hr, exercise_angina, 
                            oldpeak, st_slope]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.error(f"⚠️ ผลทำนาย: มีความเสี่ยงเป็นโรคหัวใจ (Probability: {probability[1]:.2%})")
    else:
        st.success(f"✅ ผลทำนาย: ไม่มีความเสี่ยง (Probability: {probability[0]:.2%})")