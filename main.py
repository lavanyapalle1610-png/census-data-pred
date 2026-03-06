# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("dataset.csv")

# Example: predicting income
# Target column
y = data["income"]

# Features
X = data.drop("income", axis=1)

# Convert categorical values
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Census Income Prediction")
print("Model Used: XGBoost")
print("Accuracy:", accuracy)