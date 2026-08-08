import numpy as np
import pandas as pd

def preprocess_stable(df, is_train=True, scalers=None):
    df_p = df.copy()

    df_p['person_gender'] = df_p['person_gender'].map({'male': 0, 'female': 1})
    df_p['previous_loan_defaults_on_file'] = df_p['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

    for cat in ['RENT', 'OWN', 'MORTGAGE', 'OTHER']:
        df_p[f'home_{cat}'] = (df_p['person_home_ownership'] == cat).astype(int)
    df_p = df_p.drop(columns=['person_home_ownership'])

    df_p['loan_percent_income_sq'] = df_p['loan_percent_income'] ** 2
    df_p['person_income_log'] = np.log1p(df_p['person_income'])
    df_p['loan_int_rate_sq'] = df_p['loan_int_rate'] ** 2
    df_p['risk_interaction'] = df_p['loan_percent_income'] * df_p['loan_int_rate']
    df_p['income_loan_ratio'] = df_p['loan_amnt'] / (df_p['person_income'] + 1)
    
    X_mat = df_p.values.astype(float)

    if is_train:
        mean = np.mean(X_mat, axis=0)
        std = np.std(X_mat, axis=0)
        std[std == 0] = 1
        scalers = {'mean': mean, 'std': std}

    X_mat = (X_mat - scalers['mean']) / scalers['std']

    intercept = np.ones((X_mat.shape[0], 1))
    X_mat = np.hstack((intercept, X_mat))

    return X_mat, scalers

def train_val_split(X, y, val_ratio=0.15):
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_ratio)
    return X[indices[val_size:]], X[indices[:val_size]], y[indices[val_size:]], y[indices[:val_size]]