import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("malware_dataset.csv")

# Giả sử dataset có cột 'label' + các feature là byte_0, byte_1, ...
X = df.drop("label", axis=1).values
y = df["label"].values

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train model (Random Forest đơn giản, bạn có thể đổi CNN sau)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
acc = model.score(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# 5. Save model
joblib.dump(model, "malware_model.pkl")
print("Model saved as malware_model.pkl")
