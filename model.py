import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier
import joblib
import time

# Start timer
start_time = time.time()

# Load dataset
print("Loading dataset...")
data = pd.read_csv('data.csv')  # Replace with your actual file path

# Feature engineering - simplified
print("Preprocessing data...")
def preprocess_data(df):
    # Focus on most important features
    df = df[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
             'oldbalanceDest', 'newbalanceDest', 'isFraud']].copy()
    
    # Calculate important derived features
    df['origin_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 0.01)  # Avoid division by zero
    
    # Convert type to categorical
    df['type'] = df['type'].astype('category')
    
    return df

data = preprocess_data(data)

# Split data - stratify to maintain fraud ratio
print("Splitting data...")
X = data.drop('isFraud', axis=1)
y = data['isFraud']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optimized CatBoost parameters for faster training
print("Training model...")
model = CatBoostClassifier(
    iterations=300,           # Reduced from 1000
    learning_rate=0.1,        # Increased from 0.05 for faster convergence
    depth=6,                  # Reduced from 8
    l2_leaf_reg=3,            # Regularization to prevent overfitting
    random_seed=42,
    eval_metric='AUC',
    early_stopping_rounds=20, # Stop early if no improvement
    task_type='CPU',          # Explicitly use CPU
    thread_count=-1,          # Use all available cores
    verbose=50,               # Less frequent logging
    cat_features=['type'],    # Only categorical feature
    auto_class_weights='Balanced'
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Save model
joblib.dump(model, 'optimized_fraud_model.pkl')
print(f"\nModel trained and saved in {(time.time()-start_time)/60:.1f} minutes")