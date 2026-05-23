import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction Engine")
st.write("Enter the customer's details below to predict if they will cancel their subscription.")

# Create a clean layout for inputs
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=tenure * 65.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

st.markdown("---")

# Prediction Button
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    # Package the data to send to our FastAPI backend
    # We provide default background variables to keep the UI clean
    payload = {
        "data": {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech_support,
            "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"
        }
    }

    # Send the data to the API
    with st.spinner("Analyzing customer profile..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prob = result["churn_probability"]
                
                if result["churn_prediction"] == 1:
                    st.error(f"⚠️ **HIGH RISK OF CHURN** \n\nThe model predicts this customer will cancel. (Probability: {prob}%)")
                else:
                    st.success(f"✅ **CUSTOMER IS SECURE** \n\nThe model predicts this customer will stay. (Churn Probability: {prob}%)")
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to the API. Is Uvicorn running? Error: {e}")