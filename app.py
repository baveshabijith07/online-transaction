import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

@st.cache_resource
def load_models():
    try:
        model = joblib.load('fraud_model_lgb.pkl')  # Changed to LGB model
        iso_forest = joblib.load('anomaly_model_iso.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model, iso_forest, scaler, features
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, iso_forest, scaler, features = load_models()

def main():
    st.set_page_config(page_title="Advanced Fraud Detection", layout="wide")
    st.title("💰 Advanced Transaction Fraud Detection")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount", min_value=0.01, value=150.0, step=1.0)
            transaction_type = st.selectbox(
                "Transaction Type",
                options=['CASH_IN', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']
            )
            trans_time = st.time_input("Transaction Time", value=datetime.now().time())
            is_night = 1 if trans_time.hour < 6 or trans_time.hour > 22 else 0
            
        with col2:
            orig_bal_before = st.number_input("Origin Balance Before", min_value=0.0, value=1000.0)
            orig_bal_after = st.number_input("Origin Balance After", min_value=0.0, value=850.0)
            dest_bal_before = st.number_input("Destination Balance Before", min_value=0.0, value=500.0)
            trans_last_hour = st.number_input("Transactions Last Hour", min_value=0, value=1)
            
        col3, col4 = st.columns(2)
        with col3:
            is_foreign = st.checkbox("Foreign Transaction")
        with col4:
            new_device = st.checkbox("New Device Used")
        
        submitted = st.form_submit_button("Check Fraud")
        