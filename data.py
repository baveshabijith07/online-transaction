import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_fraud_dataset(num_records=10000, fraud_rate=0.12):
    np.random.seed(42)
    
    # Essential features
    data = {
        # Core transaction info
        'timestamp': [datetime.now() - timedelta(hours=np.random.randint(0, 720)) for _ in range(num_records)],
        'transaction_type': np.random.choice(
            ['CASH_IN', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT'],
            size=num_records,
            p=[0.15, 0.2, 0.3, 0.2, 0.15]
        ),
        'amount': np.round(np.abs(np.random.normal(500, 300, num_records)), 2),
        
        # Origin account features
        'origin_balance_before': np.round(np.abs(np.random.normal(2000, 1500, num_records)), 2),
        'origin_balance_after': np.zeros(num_records),  # Initialize as array
        
        # Destination account features
        'destination_balance_before': np.round(np.abs(np.random.normal(1500, 1200, num_records)), 2),
        'destination_balance_after': np.zeros(num_records),  # Initialize as array
        
        # Behavioral features
        'transactions_last_hour': np.random.poisson(0.5, num_records),
        'is_foreign': np.random.binomial(1, 0.1, num_records),
        'new_device': np.random.binomial(1, 0.15, num_records),
        
        # Target
        'is_fraud': np.zeros(num_records)
    }
    
    # Calculate derived balances
    data['origin_balance_after'] = np.round(data['origin_balance_before'] - data['amount'], 2)
    data['destination_balance_after'] = np.round(data['destination_balance_before'] + data['amount'], 2)
    
    # Inject fraud patterns
    fraud_indices = np.random.choice(num_records, int(num_records * fraud_rate), replace=False)
    
    # Modify fraud transactions
    data['is_fraud'][fraud_indices] = 1
    data['amount'][fraud_indices] = np.round(
        np.minimum(data['amount'][fraud_indices] * np.random.uniform(3, 10, size=len(fraud_indices)), 50000), 2)
    data['transactions_last_hour'][fraud_indices] = np.minimum(
        data['transactions_last_hour'][fraud_indices] + np.random.randint(3, 10, size=len(fraud_indices)), 20)
    data['is_foreign'][fraud_indices] = np.random.binomial(1, 0.7, size=len(fraud_indices))
    data['new_device'][fraud_indices] = 1
    
    # Balance manipulation for fraud cases
    data['origin_balance_after'][fraud_indices] = np.round(
        np.random.uniform(0, 50, size=len(fraud_indices)), 2)  # Account draining
    data['destination_balance_before'][fraud_indices] = 0  # New account
    
    return pd.DataFrame(data)

# Generate and save dataset
fraud_data = generate_fraud_dataset(20000, 0.15)
fraud_data.to_csv('essential_fraud_data.csv', index=False)
print(f"Dataset generated with {fraud_data.is_fraud.sum()} fraud cases out of {len(fraud_data)} transactions")