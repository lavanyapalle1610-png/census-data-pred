# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_models(data):

    # Target variable
    y = data["income"]

    # Features
    X = data.drop("income", axis=1)

    # Convert categorical columns
    X = pd.get_dummies(X)

    # Handle missing values
    X = X.fillna(0)

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    results["Decision Tree"] = accuracy_score(y_test, dt_pred)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = accuracy_score(y_test, rf_pred)

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    results["SVM"] = accuracy_score(y_test, svm_pred)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["Logistic Regression"] = accuracy_score(y_test, lr_pred)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = accuracy_score(y_test, xgb_pred)

    return results