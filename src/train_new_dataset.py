"""
Train and evaluate a CardioPredict candidate model on the new Mendeley dataset.

Safety:
- Never overwrites the existing Cleveland model (models/cvd_model.joblib).
- Saves only to models/cvd_model_new_dataset_candidate.joblib.
- Uses patientid for traceability only; patientid is never a predictor.
- Removes rows with slope=0 because the dataset description defines slope as 1, 2, or 3.
- Treats serumcholestrol=0 as missing and imputes it inside the ML pipeline,
  so imputation is learned from training folds only.
- Keeps nonzero cholesterol values outside the published descriptive range,
  but flags them in the audit report for source verification.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Cardiovascular_Disease_Dataset_cleaned.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "cvd_model_new_dataset_candidate.joblib"
METRICS_PATH = MODEL_DIR / "cvd_metrics_new_dataset_candidate.json"

TARGET = "target"
ID_COLUMN = "patientid"

NUMERIC_FEATURES = [
    "age",
    "restingBP",
    "serumcholestrol",
    "maxheartrate",
    "oldpeak",
    "noofmajorvessels",
]
CATEGORICAL_FEATURES = [
    "gender",
    "chestpain",
    "fastingbloodsugar",
    "restingrelectro",
    "exerciseangia",
    "slope",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURES, TARGET]


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", make_preprocessor(scale_numeric=False)),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place the cleaned CSV in the project's data folder.")
        return 1

    df = pd.read_csv(DATA_PATH)
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        print("Dataset is missing required columns:", ", ".join(missing_columns))
        return 1

    # Reapply critical checks so the script remains safe if the CSV is edited.
    original_rows = len(df)
    invalid_slope_rows = int((~df["slope"].isin([1, 2, 3])).sum())
    df = df.loc[df["slope"].isin([1, 2, 3])].copy()

    zero_cholesterol = int((df["serumcholestrol"] == 0).sum())
    df["serumcholestrol"] = df["serumcholestrol"].replace(0, np.nan)

    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        print("Target must contain only non-missing binary values 0 and 1.")
        return 1
    if len(df) < 100 or df[TARGET].nunique() != 2:
        print("Not enough valid records or only one target class remains.")
        return 1

    # patientid is deliberately excluded from model inputs.
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_models()
    cv_results = {}
    fitted_on_train = {}

    print(f"Records available for training: {len(df)} (from {original_rows} rows)")
    print(f"Rows removed for invalid slope code: {invalid_slope_rows}")
    print(f"Zero cholesterol values treated as missing: {zero_cholesterol}")
    print("Comparing candidate models with 5-fold stratified CV...")

    for name, pipeline in candidates.items():
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        cv_results[name] = {
            "roc_auc_mean": float(np.mean(scores)),
            "roc_auc_std": float(np.std(scores)),
            "fold_scores": [float(x) for x in scores],
        }
        fitted_on_train[name] = pipeline.fit(X_train, y_train)
        print(
            f"{name}: CV ROC-AUC {np.mean(scores):.4f} "
            f"+/- {np.std(scores):.4f}"
        )

    selected_name = max(
        cv_results,
        key=lambda name: cv_results[name]["roc_auc_mean"],
    )
    selected_train_model = fitted_on_train[selected_name]

    y_pred = selected_train_model.predict(X_test)
    y_probability = selected_train_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "confusion_matrix_labels": ["actual_0", "actual_1"],
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_records": int(len(y_test)),
    }

    # After one held-out evaluation, refit the selected pipeline on all available
    # records for a candidate artifact. Existing project model files are untouched.
    final_candidate = build_models()[selected_name]
    final_candidate.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_candidate, MODEL_PATH)

    out_of_doc_range = int(
        (
            df["serumcholestrol"].notna()
            & ~df["serumcholestrol"].between(126, 564)
        ).sum()
    )
    payload = {
        "project": "CardioPredict",
        "dataset": "Cardiovascular_Disease_Dataset",
        "dataset_source": "Doppala & Bhattacharyya (2021), Mendeley Data, DOI 10.17632/dzz48mvjht.1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "records_after_slope_validation": int(len(df)),
        "target_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "features": FEATURES,
        "excluded_identifier": ID_COLUMN,
        "preprocessing": {
            "slope": "Rows with slope values outside {1, 2, 3} excluded",
            "serumcholestrol": "Zero values treated as missing; median imputation occurs inside each training fold",
            "cholesterol_outside_documented_range": {
                "count_retained_for_review": out_of_doc_range,
                "range_in_dataset_description": [126, 564],
            },
            "categorical_encoding": "One-hot encoding",
        },
        "cv_model_comparison": cv_results,
        "selected_model": selected_name,
        "held_out_test_metrics": test_metrics,
        "model_file": str(MODEL_PATH.relative_to(ROOT)),
        "intended_use": "Educational prototype only; not for diagnosis or clinical decisions.",
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSelected candidate:", selected_name)
    print("Held-out test metrics:")
    for metric in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"  {metric}: {test_metrics[metric]:.4f}")
    print("\nCandidate model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("The existing models/cvd_model.joblib was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
