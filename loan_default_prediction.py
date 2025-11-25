import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("synthetic_dataset_1000.csv")

print("First 5 rows:")
print(df.head())

# Drop non-useful or unique columns
df = df.drop(columns=["customer_id", "signup_date"])

# Encode categorical columns
label_cols = ["home_ownership", "education", "marital_status", "region"]
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop(columns=["target_default_risk"])
y = df["target_default_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

