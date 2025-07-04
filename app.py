import streamlit as st
import pandas as pd
import joblib

# Load model and expected columns
model = joblib.load("heart_disease_rf_improved.pkl")
model_columns = joblib.load("model_columns.pkl")  # <-- REQUIRED

st.title("💓 Heart Disease Risk Predictor")
st.markdown("Enter patient information to assess their risk of developing heart disease.")

# Input form
with st.form("input_form"):
    st.header("Patient Information")

    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    physical_health = st.slider("Physical Health (sick days in past 30)", 0, 30, 0)
    mental_health = st.slider("Mental Health (stress days in past 30)", 0, 30, 0)
    sleep_time = st.slider("Sleep Time (hours/night)", 0, 24, 7)

    col1, col2 = st.columns(2)
    with col1:
        smoking = st.selectbox("Smokes?", ["no", "yes"])
        alcohol = st.selectbox("Drinks alcohol heavily?", ["no", "yes"])
        stroke = st.selectbox("Ever had a stroke?", ["no", "yes"])
        diff_walking = st.selectbox("Has difficulty walking?", ["no", "yes"])
        physical_activity = st.selectbox("Physically active?", ["no", "yes"])

    with col2:
        sex = st.selectbox("Sex", ["female", "male"])
        diabetic = st.selectbox("Diabetic?", ["no", "yes", "yes (during pregnancy)", "borderline", "don't know"])
        asthma = st.selectbox("Has asthma?", ["no", "yes"])
        kidney_disease = st.selectbox("Kidney disease?", ["no", "yes"])
        skin_cancer = st.selectbox("Skin cancer?", ["no", "yes"])

    age = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])

    race = st.selectbox("Race", [
        'white', 'black', 'asian', 'american indian/alaskan native', 'hispanic', 'other'
    ])

    gen_health = st.selectbox("General Health", [
        'excellent', 'very good', 'good', 'fair', 'poor'
    ])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': 1 if smoking == 'yes' else 0,
        'AlcoholDrinking': 1 if alcohol == 'yes' else 0,
        'Stroke': 1 if stroke == 'yes' else 0,
        'DiffWalking': 1 if diff_walking == 'yes' else 0,
        'Sex': 1 if sex == 'male' else 0,
        'Diabetic': 1 if diabetic in ['yes', 'yes (during pregnancy)', 'borderline', 'no, borderline diabetes'] else 0,
        'PhysicalActivity': 1 if physical_activity == 'yes' else 0,
        'Asthma': 1 if asthma == 'yes' else 0,
        'KidneyDisease': 1 if kidney_disease == 'yes' else 0,
        'SkinCancer': 1 if skin_cancer == 'yes' else 0,
    }

    # One-hot encoding
    age_categories = [
        'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44',
        'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59', 'AgeCategory_60-64',
        'AgeCategory_65-69', 'AgeCategory_70-74', 'AgeCategory_75-79', 'AgeCategory_80 or older'
    ]
    for cat in age_categories:
        input_dict[cat] = 1 if age in cat else 0

    race_categories = [
        'Race_white', 'Race_black', 'Race_asian', 'Race_american indian/alaskan native',
        'Race_hispanic', 'Race_other'
    ]
    for r in race_categories:
        input_dict[r] = 1 if race in r else 0

    gen_categories = [
        'GenHealth_excellent', 'GenHealth_fair', 'GenHealth_good',
        'GenHealth_poor', 'GenHealth_very good'
    ]
    for g in gen_categories:
        input_dict[g] = 1 if gen_health in g else 0

    # Final input DataFrame with correct columns
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    confidence = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🚨 Patient is AT RISK of heart disease. Confidence: {confidence:.2f}")
    else:
        st.success(f"✅ Patient is NOT at risk of heart disease. Confidence: {1 - confidence:.2f}")
