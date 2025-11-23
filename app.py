import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="Heart Disease App",
    page_icon="🫀",
    layout="centered",
    menu_items={
        'about': "Heart Disease Prediction App – by Hexacore Team"
    }
)


st.title("🫀 Heart Disease Prediction App")
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
    thal = st.selectbox("Thal (1-normal, 2-fixed defect, 3-reversible)", [1, 2, 3])

sex = 1 if sex == "male" else 0

# Order of columns based on training
# Create base feature vector (13 features)
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]], dtype=float)

# Match the number of features expected by the model
required_features = getattr(model, "n_features_in_", input_data.shape[1])

if input_data.shape[1] < required_features:
    # Pad with zeros to reach the required number of features
    padded = np.zeros((input_data.shape[0], required_features))
    padded[:, :input_data.shape[1]] = input_data
    X = padded
elif input_data.shape[1] > required_features:
    # (Just in case) cut extra features
    X = input_data[:, :required_features]
else:
    X = input_data

if st.button("🔍 Predict"):
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease**\nProbability: **{probability:.2f}**")
    else:
        st.success(f"💚 **No Heart Disease Detected**\nProbability: **{probability:.2f}**")

st.markdown("---")
st.caption("Made by Hexacore — IIT Ropar | FDS Project")
