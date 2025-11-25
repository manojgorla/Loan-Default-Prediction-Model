import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from features import create_features   # ← import here

# Wrap feature engineering
feature_engineer = FunctionTransformer(create_features)

# Example dataset
data = pd.DataFrame({
    "income": [50000, 30000, 40000, 60000, 25000, 28000, 22000, 100000],
    "loan_amount": [20000, 15000, 12000, 25000, 20000, 18000, 15000, 20000],
    "credit_score": [700, 650, 680, 720, 620, 630, 600, 750],
    "loan_term_months": [12, 6, 9, 24, 6, 12, 6, 10],
    "default": [0, 1, 0, 0, 1, 1, 1, 0]
})

X = data.drop("default", axis=1)
y = data["default"]

pipeline = Pipeline([
    ("features", feature_engineer),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "loan_default_model.pkl")
print("✅ Model saved as loan_default_model.pkl")