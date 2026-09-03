"""Reusable data loading, cleaning, feature engineering, and preprocessing."""

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
TARGET = "Churn"
RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", TARGET,
]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "tenure_years", "avg_monthly_charge"]
CATEGORICAL_FEATURES = [column for column in RAW_COLUMNS if column not in {"customerID", TARGET, *NUMERIC_FEATURES}]


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load a local copy when present, otherwise download the public IBM dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    local_path = data_dir / "Telco-Customer-Churn.csv"
    if not local_path.exists():
        frame = pd.read_csv(DATA_URL)
        frame.to_csv(local_path, index=False)
    return pd.read_csv(local_path)


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic cleaning and return audit information."""
    data = frame.copy()
    data.columns = data.columns.str.strip()
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET].isin(["Yes", "No"])].reset_index(drop=True)
    audit = {
        "rows_before": int(len(frame)),
        "rows_after": int(len(data)),
        "duplicate_rows_removed": duplicate_rows,
        "missing_values_before_pipeline": {k: int(v) for k, v in data.isna().sum().items() if v},
    }
    return data, audit


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add features using only columns present in a new customer record."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
        data["tenure_years"] = pd.to_numeric(data["tenure"], errors="coerce") / 12
        data["avg_monthly_charge"] = pd.to_numeric(
            data["TotalCharges"] / data["tenure"].replace(0, pd.NA), errors="coerce"
        )
        data = data.drop(columns=["customerID", TARGET], errors="ignore")
        return data


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])