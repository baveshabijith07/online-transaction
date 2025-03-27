import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess data
def load_data():
    df = pd.read_csv('data.csv')
    
    # Feature engineering
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df['amount_to_balance_ratio'] = df['amount'] / (df['origin_balance_before'] + 1)
    df['balance_discrepancy'] = abs((df['origin_balance_before'] - df['amount']) - df['origin_balance_after'])
    
    return df

# Train model
def train_model():
    df = load_data()
    
    # Select features
    features = [
        'amount', 'origin_balance_before', 'origin_balance_after',
        'destination_balance_before', 'transactions_last_hour',
        'is_foreign', 'new_device', 'is_night', 'amount_to_balance_ratio',
        'balance_discrepancy'
    ]
    
    X = df[features]
    y = df['is_fraud']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Train Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    iso_forest.fit(X_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))
    
    # Save models
    joblib.dump(model, 'fraud_model_xgb.pkl')
    joblib.dump(iso_forest, 'anomaly_model_iso.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(features, 'features.pkl')

if __name__ == '__main__':
    train_model()