import streamlit as st
import pandas as pd
import joblib

# Load model with caching
@st.cache_resource
def load_model():
    try:
        return joblib.load('optimized_fraud_model.pkl')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

def main():
    st.title("Online Transaction Fraud Detection System")
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Hours since start (step)", min_value=0)
            amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
            transaction_type = st.selectbox(
                "Transaction Type",
                options=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
            )
            
        with col2:
            oldbalanceOrg = st.number_input("Origin balance before", min_value=0.0, format="%.2f")
            newbalanceOrig = st.number_input("Origin balance after", min_value=0.0, format="%.2f")
            oldbalanceDest = st.number_input("Destination balance before", min_value=0.0, format="%.2f")
            newbalanceDest = st.number_input("Destination balance after", min_value=0.0, format="%.2f")
        
        submitted = st.form_submit_button("Check for Fraud")
    
    if submitted:
        # Calculate balance changes
        origin_change = oldbalanceOrg - newbalanceOrig
        dest_change = newbalanceDest - oldbalanceDest
        balance_ratio = amount / (oldbalanceOrg + 0.01)  # Avoid division by zero
        
        input_data = {
            'step': step,
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest,
            'origin_balance_change': origin_change,
            'dest_balance_change': dest_change,
            'balance_ratio': balance_ratio
        }
        
        try:
            input_df = pd.DataFrame([input_data])
            proba = model.predict_proba(input_df)[0]
            fraud_prob = proba[1] * 100
            
            st.subheader("Fraud Detection Result")
            
            if fraud_prob > 50:  # Using 50% as threshold
                st.error(f"⚠️ High Fraud Risk: {fraud_prob:.1f}%")
                st.progress(fraud_prob/100)
                
                # Explain risk factors
                risk_factors = []
                if transaction_type in ['TRANSFER', 'CASH_OUT']:
                    risk_factors.append(f"High-risk transaction type: {transaction_type}")
                if origin_change != amount:
                    risk_factors.append(f"Amount ({amount}) doesn't match origin balance change ({origin_change})")
                if newbalanceOrig < 10 and oldbalanceOrg > 100:
                    risk_factors.append("Origin account nearly emptied")
                
                if risk_factors:
                    st.write("Detected risk factors:")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                
                st.warning("Recommendation: Manually verify this transaction")
            else:
                st.success(f"✅ Low Fraud Risk: {fraud_prob:.1f}%")
                st.progress(fraud_prob/100)
                st.write("No significant fraud indicators detected")
            
            # Show balance changes
            with st.expander("View Balance Changes"):
                st.write(f"Origin balance change: {origin_change:.2f}")
                st.write(f"Destination balance change: {dest_change:.2f}")
                st.write(f"Amount to origin balance ratio: {balance_ratio:.2f}")
        
        except Exception as e:
            st.error(f"Error analyzing transaction: {str(e)}")

if __name__ == '__main__':
    main()