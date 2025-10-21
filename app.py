import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("logistic_columns_heart.pkl")
    return model, scaler, columns

model_, scaler_, expected_columns = load_artifacts()

# Streamlit UI
st.title("❤️ HEART STROKE PREDICTION BY GAURAV SINGH BISHT 😍😁")
st.markdown("Provide the following details:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ['Male', 'Female'])
chestpain = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
resting_bp = st.slider("Resting BP (mm Hg)", 0, 200, 120)
cholesterol = st.slider("Cholesterol (mm Hg)", 0, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", ['0', '1'])
resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
max_hr = st.slider("Maximum Heart Rate", 0, 200, 100)
exercise_angina = st.selectbox("Exercise Induced Angina", ['No', 'Yes'])
oldpeak = st.slider("ST depression (Oldpeak)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("Slope of the peak exercise ST segment", ['Upsloping', 'Flat', 'Downsloping'])

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chestpain,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler_.transform(input_df)
    prediction = model_.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
