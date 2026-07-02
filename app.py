import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Seeds Classification",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Seeds Classification using Machine Learning")
st.write("Enter the seed measurements below and click Predict.")

# -----------------------------
# User Inputs
# -----------------------------

area = st.number_input("Area", min_value=0.0, value=15.26)

perimeter = st.number_input("Perimeter", min_value=0.0, value=14.84)

compactness = st.number_input("Compactness", min_value=0.0, value=0.87)

kernel_length = st.number_input("Kernel Length", min_value=0.0, value=5.76)

kernel_width = st.number_input("Kernel Width", min_value=0.0, value=3.31)

asymmetry = st.number_input("Asymmetry Coefficient", min_value=0.0, value=2.22)

groove_length = st.number_input("Kernel Groove Length", min_value=0.0, value=5.22)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    sample = pd.DataFrame([{
        "Area": area,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "Kernel_Length": kernel_length,
        "Kernel_Width": kernel_width,
        "Asymmetry_Coefficient": asymmetry,
        "Kernel_Groove_Length": groove_length
    }])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    if prediction == 1:
        st.success("🌾 Predicted Class: 1 (Kama Wheat)")

    elif prediction == 2:
        st.success("🌾 Predicted Class: 2 (Rosa Wheat)")

    else:
        st.success("🌾 Predicted Class: 3 (Canadian Wheat)")