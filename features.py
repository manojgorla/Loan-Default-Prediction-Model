# features.py
def create_features(df):
    df = df.copy()
    df['debt_to_income'] = df['loan_amount'] / df['income']
    df['monthly_income'] = df['income'] / 12
    df['monthly_payment'] = df['loan_amount'] / df['loan_term_months']
    df['payment_to_income'] = df['monthly_payment'] / df['monthly_income']
    df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
    return df