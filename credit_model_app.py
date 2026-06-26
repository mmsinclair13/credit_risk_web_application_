import joblib
import pandas as pd
import streamlit as st

model = joblib.load("random_forest_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.plk") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction Application")
st.write("Enter the application information to predict if the credit risk is good or bad")

Age = st.number_input("Age", min_value = 18, max_value = 90, value = 30)
Sex = st.selectbox("Sex", ["male", "female"])
Job = st.number_input("Job (0-3)", min_value = 0, max_value = 3, value =1)
Checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
Saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
Credit_amount = st.number_input("Credit amount", min_value = 0, value = 100)
Duration = st.number_input("Duration", min_value =1, value =12)
Housing = st.selectbox("Housing", ["own", "rent", "free"])

input_df = pd.DataFrame({
    "Age": [Age],
    "Sex": [encoders["Sex"].transform([Sex])[0]],
    "Job": [Job],
    "Checking account": [encoders["Checking account"].transform([Checking_account])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([Saving_accounts])[0]],
    "Credit amount": [Credit_amount],
    "Duration": [Duration],
    "Housing": [encoders["Housing"].transform([Housing])[0]],
})

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")