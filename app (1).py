import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------
# LOAD MODEL + ENCODERS + COLUMNS
# ------------------------------
model = joblib.load("sleep_rf_model.pkl")
encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Sleep Disorder Prediction - Health Scout AI", page_icon="😴")

# ------------------------------
# APP HEADER
# ------------------------------
st.title("🧠 Health Scout AI")
st.subheader("Sleep Disorder Prediction System")
st.write("Enter your lifestyle & health information to predict your likelihood of **Insomnia** or **Sleep Apnea**.")

# ------------------------------
# USER INPUT FORM
# ------------------------------
st.header("📋 Input Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    occupation = st.selectbox("Occupation", [
        "Software Engineer", "Doctor", "Nurse", "Teacher", 
        "Sales Representative", "Manager", "Others"
    ])
    bmi = st.selectbox("BMI Category", ["Underweight", "Normal Weight", "Overweight", "Obese"])

with col2:
    sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=12.0, value=6.0)
    quality = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
    activity = st.number_input("Physical Activity (minutes/day)", min_value=0, max_value=300, value=30)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=75)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=5000)

bp_systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
bp_diastolic = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)

# ------------------------------
# PREPROCESS USER INPUT
# ------------------------------
if st.button("Predict Sleep Disorder"):

    # Create input dataframe
    input_data = pd.DataFrame([[gender, age, occupation, sleep_duration, quality,
                                activity, stress, bmi, heart_rate, daily_steps,
                                bp_systolic, bp_diastolic]], 
                              columns=feature_columns)

    # Encode categorical columns
    for col in ["Gender", "Occupation", "BMI Category"]:
        input_data[col] = encoders[col].transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    # Decode prediction
    disorder_label = encoders["Sleep Disorder"].inverse_transform([prediction])[0]

    st.subheader("🧾 Prediction Result")
    
    if disorder_label == "Insomnia":
        st.error(f"🔴 High likelihood of **Insomnia** ({probs[prediction]*100:.1f}% confidence)")
    elif disorder_label == "Sleep Apnea":
        st.warning(f"🟠 Likely **Sleep Apnea** ({probs[prediction]*100:.1f}% confidence)")
    else:
        st.success(f"🟢 No Sleep Disorder Detected ({probs[prediction]*100:.1f}% confidence)")

    st.write("⚠️ *This prediction is not a medical diagnosis. Consult a professional for confirmation.*")
