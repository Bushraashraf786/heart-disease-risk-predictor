import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="CardioCheck AI", page_icon=":anatomical_heart:", layout="centered")

model = joblib.load("heart_disease_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("CardioCheck AI")
st.write("Heart Disease Risk Predictor - fill in the patient details below.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                                3: "Non-anginal pain", 4: "Asymptomatic"}[x]
    )
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", ["Yes", "No"])
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality",
                                2: "Left ventricular hypertrophy"}[x]
    )

with col2:
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina?", ["Yes", "No"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x]
    )
    ca = st.slider("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.selectbox(
        "Thalassemia",
        options=[3, 6, 7],
        format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x]
    )

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    values = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg,
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    ordered_input = [values[feat] for feat in feature_names]
    X = np.array(ordered_input).reshape(1, -1)

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = proba[prediction] * 100

    if prediction == 1:
        st.error(f"High risk of heart disease (confidence: {confidence:.1f}%)")
    else:
        st.success(f"Low risk of heart disease (confidence: {confidence:.1f}%)")

    st.caption("This is a machine learning estimate, not a medical diagnosis. Please consult a doctor for any real health concerns.")
