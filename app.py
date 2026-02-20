import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# ---------------- UI ---------------- #

st.set_page_config(page_title="NovaGen AI", layout="centered")

st.title("🧬 NovaGen AI - High Risk Detection System")
st.markdown("### Predict whether a patient is High Risk")

st.markdown("---")

st.sidebar.header("📝 Enter Patient Details")

# Replace these names with your real dataset columns
feature1 = st.sidebar.number_input("BMI", min_value=0.0)
feature2 = st.sidebar.number_input("Blood_Pressure", min_value=0.0)
feature3 = st.sidebar.number_input("Cholesterol", min_value=0.0)
feature4 = st.sidebar.number_input("Stress_Level", min_value=0.0)

st.markdown("---")

# ---------------- Prediction ---------------- #

if st.sidebar.button("🚀 Predict Risk"):

    input_data = np.array([[feature1, feature2, feature3, feature4]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk Patient Detected")
        st.write("Please consult a medical professional immediately.")
    else:
        st.success("✅ Patient is Safe")
        st.write("No immediate high-risk indicators detected.")
