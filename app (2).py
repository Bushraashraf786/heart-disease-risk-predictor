import gradio as gr
import joblib
import numpy as np

# Load the trained model and the exact feature order it was trained on
model = joblib.load("heart_disease_model.pkl")
feature_names = joblib.load("feature_names.pkl")


def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal):
    # Build the feature vector in the EXACT order the model expects.
    # This dict lets us map by name instead of relying on argument order,
    # so it can never get scrambled.
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
        result = f"⚠️ **High risk of heart disease** (confidence: {confidence:.1f}%)"
    else:
        result = f"✅ **Low risk of heart disease** (confidence: {confidence:.1f}%)"

    result += "\n\n_This is a machine learning estimate, not a medical diagnosis. Please consult a doctor for any real health concerns._"
    return result


with gr.Blocks(title="CardioCheck AI - Heart Disease Risk Predictor") as demo:
    gr.Markdown("# ⚡ CardioCheck AI: Heart Disease Risk Predictor")
    gr.Markdown("Enter patient details below and click **Predict** to estimate heart disease risk.")

    with gr.Row():
        with gr.Column():
            age = gr.Slider(20, 100, value=50, step=1, label="Age (years)")
            sex = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            cp = gr.Dropdown(
                choices=[("Typical angina", 1), ("Atypical angina", 2),
                         ("Non-anginal pain", 3), ("Asymptomatic", 4)],
                value=1, label="Chest Pain Type"
            )
            trestbps = gr.Slider(80, 220, value=120, label="Resting Blood Pressure (mm Hg)")
            chol = gr.Slider(100, 600, value=200, label="Serum Cholesterol (mg/dl)")
            fbs = gr.Radio(["Yes", "No"], value="No", label="Fasting Blood Sugar > 120 mg/dl")
            restecg = gr.Dropdown(
                choices=[("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)],
                value=0, label="Resting ECG Results"
            )

        with gr.Column():
            thalach = gr.Slider(60, 220, value=150, label="Max Heart Rate Achieved")
            exang = gr.Radio(["Yes", "No"], value="No", label="Exercise Induced Angina")
            oldpeak = gr.Slider(0.0, 7.0, value=1.0, step=0.1, label="ST Depression (oldpeak)")
            slope = gr.Dropdown(
                choices=[("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)],
                value=1, label="Slope of Peak Exercise ST Segment"
            )
            ca = gr.Slider(0, 4, value=0, step=1, label="Number of Major Vessels (0-4)")
            thal = gr.Dropdown(
                choices=[("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)],
                value=3, label="Thalassemia"
            )

    predict_btn = gr.Button("🔍 Predict", variant="primary")
    output = gr.Markdown()

    predict_btn.click(
        fn=predict_heart_disease,
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
