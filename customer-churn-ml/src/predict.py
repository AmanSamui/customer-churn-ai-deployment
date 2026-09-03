"""Make a prediction from the saved end-to-end pipeline."""

from pathlib import Path
import sys

import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import FeatureEngineer  # noqa: F401 - registers the serialized transformer


def predict_customer(customer: dict) -> dict:
    pipeline = load(ROOT / "model" / "churn_pipeline.joblib")
    frame = pd.DataFrame([customer])
    prediction = pipeline.predict(frame)[0]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    return {"churn_prediction": prediction, "churn_probability": round(probability, 4)}


if __name__ == "__main__":
    sample = {
        "customerID": "DEMO-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 53.85, "TotalCharges": "108.15",
    }
    print(predict_customer(sample))