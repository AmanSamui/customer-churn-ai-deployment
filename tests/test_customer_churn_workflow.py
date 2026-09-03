"""Regression checks for the model-building workflow and saved artifact."""

import json
import sys
from pathlib import Path

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1] / "customer-churn-ml"
sys.path.insert(0, str(ROOT))


def test_reports_and_artifact_exist_and_have_requested_metrics():
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    assert {row["model"] for row in metrics["models"]} == {"logistic_regression", "random_forest"}
    assert all(set(("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")) <= row.keys()
               for row in metrics["models"])
    assert (ROOT / "reports/model_comparison.csv").exists()
    assert (ROOT / "reports/logistic_regression_confusion_matrix.png").exists()
    assert (ROOT / "reports/random_forest_confusion_matrix.png").exists()
    assert (ROOT / "model/churn_pipeline.joblib").exists()


def test_saved_pipeline_reloads_and_predicts_from_project_root():
    pipeline = load(ROOT / "model/churn_pipeline.joblib")
    customer = {
        "customerID": "TEST-0001", "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 53.85,
        "TotalCharges": "108.15",
    }
    frame = pd.DataFrame([customer])
    assert pipeline.predict(frame)[0] in {"Yes", "No"}
    assert 0 <= float(pipeline.predict_proba(frame)[0, 1]) <= 1


def test_cleaning_removes_duplicates_and_converts_blank_total_charges():
    from src.preprocessing import clean_dataset

    frame = pd.DataFrame([
        {"customerID": "a", "TotalCharges": " ", "Churn": " Yes "},
        {"customerID": "a", "TotalCharges": " ", "Churn": " Yes "},
    ])
    cleaned, audit = clean_dataset(frame)
    assert len(cleaned) == 1
    assert pd.isna(cleaned.loc[0, "TotalCharges"])
    assert audit["duplicate_rows_removed"] == 1
