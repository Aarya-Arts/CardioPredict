import os
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
CATEGORICAL = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
NUMERIC = ["age","trestbps","chol","thalach","oldpeak"]
FEATURES = NUMERIC + CATEGORICAL

def load_dataset():
    os.makedirs("data", exist_ok=True)
    path = "data/processed.cleveland.data"
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["target"])
    # UCI target: 0 = no disease; 1..4 = disease present.
    df["target"] = (df["target"] > 0).astype(int)
    return df

def build_pipeline(model):
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])
    return Pipeline([("preprocess", prep), ("classifier", model)])

def main():
    df = load_dataset()
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }
    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
        }
        fitted[name] = pipe
    # Select model by F1 score; this is a transparent academic selection rule.
    best_name = max(results, key=lambda n: results[n]["f1_score"])
    os.makedirs("models", exist_ok=True)
    joblib.dump(fitted[best_name], "models/cvd_model.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump({"selected_model": best_name, "results": results}, f, indent=2)
    print("Training completed.")
    print("Selected model:", best_name)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
