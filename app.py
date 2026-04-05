import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model
model = joblib.load("model.pkl")

# Load dataset (for visualization)
df = pd.read_csv("novagen_dataset.csv")

# ---------------- UI ---------------- #

st.set_page_config(page_title="NovaGen AI", layout="centered")

st.title("🧬 NovaGen AI - High Risk Detection System")

st.markdown("""
## 📌 About This Project
This AI system predicts high-risk patients using Machine Learning.  
We selected Random Forest because it achieved highest recall.
""")

st.markdown("### Predict whether a patient is High Risk")
st.markdown("---")

st.sidebar.header("📝 Enter Patient Details")

# ---------------- ALL FEATURES (UNCHANGED) ---------------- #

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

    input_data = np.array([[feature1, feature2, feature3, feature4, feature5,
                            feature6, feature7, feature8, feature9, feature10,
                            feature11, feature12, feature13, feature14, feature15,
                            feature16, feature17, feature18, feature19, feature20,
                            feature21, feature22]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk Patient Detected")
        st.write("Please consult a medical professional immediately.")
    else:
        st.success("✅ Patient is Safe")
        st.write("No immediate high-risk indicators detected.")

# ---------------- VISUALIZATION ---------------- #

st.markdown("---")
st.subheader("📊 Data Visualization Dashboard")

if st.checkbox("Show Visualizations"):

    # 1️⃣ Target Distribution
    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax)
    st.pyplot(fig)

    # 2️⃣ Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 3️⃣ Age Distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], kde=True, ax=ax)
    st.pyplot(fig)

    # 4️⃣ BMI Distribution
    st.write("### BMI Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["BMI"], kde=True, ax=ax)
    st.pyplot(fig)

    # 5️⃣ Stress Level vs Target
    st.write("### Stress Level vs Risk")
    fig, ax = plt.subplots()
    sns.boxplot(x="Target", y="Stress_Level", data=df, ax=ax)
    st.pyplot(fig)

    # 6️⃣ Feature Importance (from model)
    st.write("### Feature Importance")

    importances = model.feature_importances_
    features = df.drop("Target", axis=1).columns

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    st.pyplot(fig)