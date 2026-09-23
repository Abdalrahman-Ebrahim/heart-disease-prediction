import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/heart_disease_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application uses a Machine Learning model (Logistic Regression) to predict the likelihood of heart disease 
based on patient clinical parameters. Please enter the clinical metrics below.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal & Basic Vital Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

with col2:
    st.subheader("ECG & Cardiac Test Metrics")
    cp = st.selectbox(
        "Chest Pain Type", 
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x]
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl", 
        options=[0, 1], 
        format_func=lambda x: "False" if x == 0 else "True"
    )
    restecg = st.selectbox(
        "Resting ECG Results", 
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

with col3:
    st.subheader("Exercise & Angiography Diagnostics")
    exang = st.selectbox(
        "Exercise Induced Angina", 
        options=[0, 1], 
        format_func=lambda x: "No" if x == 0 else "Yes"
    )
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment", 
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )
    ca = st.selectbox("Number of Major Vessels (0-4) Colored by Flourosopy", options=[0, 1, 2, 3, 4])
    thal = st.selectbox(
        "Thalassemia Status", 
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Null / Unknown",
            1: "Normal",
            2: "Fixed Defect",
            3: "Reversable Defect"
        }[x]
    )

st.divider()

if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    input_features = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)[0]
    prediction_proba = model.predict_proba(scaled_features)[0]

    st.subheader("Prediction Result")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("⚠️ High Risk: The model predicts a high probability of Heart Disease.")
        else:
            st.success("✅ Low Risk: The model predicts a low probability of Heart Disease.")
            
    with res_col2:
        st.metric(label="Probability of Disease", value=f"{prediction_proba[1] * 100:.2f}%")
        st.metric(label="Probability of Healthy", value=f"{prediction_proba[0] * 100:.2f}%")