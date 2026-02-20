import streamlit as st
import numpy as np

# Load trained model
import joblib
model = joblib.load("model.pkl")

# ---------------- UI ---------------- #

st.set_page_config(page_title="NovaGen AI", layout="centered")

st.title("🧬 NovaGen AI - High Risk Detection System")
st.markdown("### Predict whether a patient is High Risk")

st.markdown("---")

st.sidebar.header("📝 Enter Patient Details")

# Replace these names with your real dataset columns
feature1 = st.sidebar.number_input("Age", min_value=0.0)
feature2 = st.sidebar.number_input("BMI", min_value=0.0)
feature3 = st.sidebar.number_input("Blood_Pressure", min_value=0.0)
feature4 = st.sidebar.number_input("Cholesterol", min_value=0.0)
feature5 = st.sidebar.number_input("Glucose_Level", min_value=0.0)
feature6 = st.sidebar.number_input("Heart_Rate", min_value=0.0)
feature7 = st.sidebar.number_input("Sleep_Hours", min_value=0.0)
feature8 = st.sidebar.number_input("Exercise_Hours", min_value=0.0)
feature9 = st.sidebar.number_input("Water_Intake", min_value=0.0)
feature10 = st.sidebar.number_input("Stress_Level", min_value=0.0)
feature11 = st.sidebar.number_input("Smoking", min_value=0.0)
feature12 = st.sidebar.number_input("Alcohol", min_value=0.0)
feature13 = st.sidebar.number_input("Diet", min_value=0.0)
feature14 = st.sidebar.number_input("MentalHealth", min_value=0.0)
feature15 = st.sidebar.number_input("PhysicalActivity", min_value=0.0)
feature16 = st.sidebar.number_input("MedicalHistory", min_value=0.0)
feature17 = st.sidebar.number_input("Allergies", min_value=0.0)
feature18 = st.sidebar.number_input("Diet_Type__Vegan", min_value=0.0)
feature19 = st.sidebar.number_input("Diet_Type__Vegetarian", min_value=0.0)
feature20 = st.sidebar.number_input("Blood_Group_AB", min_value=0.0)
feature21 = st.sidebar.number_input("Blood_Group_B", min_value=0.0)
feature22 = st.sidebar.number_input("Blood_Group_O", min_value=0.0)
st.markdown("---")

# ---------------- Prediction ---------------- #

if st.sidebar.button("🚀 Predict Risk"):

    input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22,]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk Patient Detected")
        st.write("Please consult a medical professional immediately.")
    else:
        st.success("✅ Patient is Safe")
        st.write("No immediate high-risk indicators detected.")


