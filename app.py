import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="centered"
)

# -----------------------------
# Load model & features
# -----------------------------
model = joblib.load("credit_risk_model.pkl")
features = joblib.load("selected_features.pkl")

# -----------------------------
# Encoding maps (CRITICAL FIX)
# -----------------------------
gender_map = {"Male": 1, "Female": 2}
education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Other": 4
}
marriage_map = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}

# -----------------------------
# App UI
# -----------------------------
st.title("💳 Credit Risk Prediction App")
st.write("Predict the probability of credit card default using a tuned XGBoost model.")

# -----------------------------
# Customer Info
# -----------------------------
st.subheader("Customer Information")

LIMIT_BAL = st.number_input(
    "Credit Limit",
    min_value=10000,
    value=200000
)

AGE = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

SEX_label = st.selectbox("Gender", list(gender_map.keys()))
SEX = gender_map[SEX_label]

EDUCATION_label = st.selectbox("Education", list(education_map.keys()))
EDUCATION = education_map[EDUCATION_label]

MARRIAGE_label = st.selectbox("Marital Status", list(marriage_map.keys()))
MARRIAGE = marriage_map[MARRIAGE_label]

# -----------------------------
# Payment Behaviour
# -----------------------------
st.subheader("Payment Behaviour")

PAY_0 = st.number_input(
    "Last Month Pending Payments",
    min_value=-1,
    max_value=9,
    value=0
)

PAY_2 = st.number_input(
    "2 Months Ago Pending Payments",
    min_value=-1,
    max_value=9,
    value=0
)

# -----------------------------
# Billing Info
# -----------------------------
st.subheader("Billing Information")

BILL_AMT1 = st.number_input(
    "Last Bill Amount",
    min_value=0,
    value=50000
)

PAY_AMT1 = st.number_input(
    "Last Payment Amount",
    min_value=0,
    value=20000
)

# -----------------------------
# Create input dataframe
# -----------------------------
input_df = pd.DataFrame([{
    "LIMIT_BAL": LIMIT_BAL,
    "SEX": SEX,
    "EDUCATION": EDUCATION,
    "MARRIAGE": MARRIAGE,
    "AGE": AGE,
    "PAY_0": PAY_0,
    "PAY_2": PAY_2,
    "BILL_AMT1": BILL_AMT1,
    "PAY_AMT1": PAY_AMT1
}])

# Ensure correct feature order
input_df = input_df[features]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Default Risk"):
    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.5:
        st.error(f"⚠️ High Risk of Default (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low Risk Customer (Probability: {probability:.2%})")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Tuned XGBoost • RandomizedSearchCV • Streamlit • Project by Jeswin")
