import pytest
from fastapi.testclient import TestClient

from app.main import app, load_model

client = TestClient(app)
VALID_CUSTOMER = {
    "customerID": "API-0001", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 53.85, "TotalCharges": 108.15,
}


def setup_function():
    load_model.cache_clear()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_prediction_uses_real_artifact():
    response = client.post("/predict", json=VALID_CUSTOMER)
    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] == "Yes"
    assert body["churn_probability"] == pytest.approx(0.6467)
    assert body["risk_level"] == "Medium"


def test_invalid_input_and_missing_field():
    invalid = {**VALID_CUSTOMER, "tenure": "not-a-number"}
    assert client.post("/predict", json=invalid).status_code == 422
    missing = {key: value for key, value in VALID_CUSTOMER.items() if key != "Contract"}
    assert client.post("/predict", json=missing).status_code == 422


def test_invalid_categorical_value():
    invalid = {**VALID_CUSTOMER, "Contract": "Weekly"}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert "Contract" in response.text


def test_model_loading_failure(monkeypatch):
    def fail_loading():
        raise OSError("test model unavailable")

    monkeypatch.setattr("app.main.load_model", fail_loading)
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 503
    assert "Model loading failed" in response.json()["detail"]


def test_prediction_failure(monkeypatch):
    class BrokenModel:
        def predict(self, frame):
            raise ValueError("test prediction failure")

    monkeypatch.setattr("app.main.load_model", lambda: BrokenModel())
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]


def test_predict_file():
    import io
    import pandas as pd
    df = pd.DataFrame([VALID_CUSTOMER])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/predict/file",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["prediction"] == "Yes"
    assert body[0]["churn_probability"] == pytest.approx(0.6467)