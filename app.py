import streamlit as st
import pandas as pd
import joblib
from features import create_features   # ← import here

# Load pipeline
pipeline = joblib.load("loan_default_model.pkl")

st.title("💳 Loan Default Prediction App")

income = st.number_input("Income", min_value=0, step=1000)
loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
credit_score = st.number_input("Credit Score", min_value=0, max_value=850, step=10)
loan_term_months = st.number_input("Loan Term (months)", min_value=0, step=1)

user_data = pd.DataFrame({
    "income": [income],
    "loan_amount": [loan_amount],
    "credit_score": [credit_score],
    "loan_term_months": [loan_term_months]
})

if st.button("Predict"):
    prediction = pipeline.predict(user_data)[0]
    # Flip logic: 1 = applicable, 0 = not applicable
    loan_applicable = 1 if prediction == 0 else 0
    st.write("Loan Applicability:", loan_applicable)