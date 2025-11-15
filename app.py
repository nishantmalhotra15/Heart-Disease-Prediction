import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below and click Predict to check heart disease risk.")

# ---------- USER INPUTS ----------
st.header("🔧 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=90, value=50)
    sex = st.radio("Sex", ["male", "female"])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 130)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 202, 150)
    exang = st.radio("Exercise-Induced Angina (exang)", [0, 1])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0)
    slope = st.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3])

# Convert sex to numeric
sex = 1 if sex == "male" else 0

# --------- MATCH EXACT FEATURE ORDER ---------
# ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

# Scale according to training scaler
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**\nProbability: **{probability:.2f}**")
    else:
        st.success(f"💚 **No Heart Disease Detected**\nProbability: **{probability:.2f}**")

st.markdown("---")
st.caption("Made by Hexacore — IIT Ropar | FDS Project")

