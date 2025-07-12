import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")

with open("model.pkl", "rb") as f:
    model, encoders = pickle.load(f)

gender = st.selectbox("Gender", encoders["Gender"].classes_)
married = st.selectbox("Married", encoders["Married"].classes_)
dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)
education = st.selectbox("Education", encoders["Education"].classes_)
self_employed = st.selectbox("Self Employed", encoders["Self_Employed"].classes_)
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in days)", value=360)
credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1.0, 0.0])
property_area = st.selectbox("Property Area", encoders["Property_Area"].classes_)

if st.button("Predict Loan Approval"):
    input_data = np.array([[
        encoders["Gender"].transform([gender])[0],
        encoders["Married"].transform([married])[0],
        encoders["Dependents"].transform([dependents])[0],
        encoders["Education"].transform([education])[0],
        encoders["Self_Employed"].transform([self_employed])[0],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        encoders["Property_Area"].transform([property_area])[0],
    ]])

    prediction = model.predict(input_data)[0]
    st.success(f"✅ Loan Status: {'Approved' if prediction == 1 else 'Not Approved'}")