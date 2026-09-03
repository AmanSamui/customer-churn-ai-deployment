# Customer Churn Prediction ML Project

A clean, reproducible machine-learning project for predicting telecom customer churn, now with a minimal Phase 2 FastAPI backend. The saved model is not retrained or modified by the API.

## Business problem

Telecom providers lose revenue when customers leave. A churn probability can help customer-success teams prioritize retention outreach. Recall matters because missing a customer who is likely to churn can be costly, while precision matters because retention incentives are not free. Therefore, accuracy is reported but is not used alone to select the model.

## Dataset

This project uses the public [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d), downloaded from its raw CSV source by `src/preprocessing.py` and cached in `data/` after the first run. It contains 7,043 customer records and the target `Churn` (`Yes`/`No`). `customerID` is an identifier and is excluded from modeling.

Columns include demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account history (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`), services (`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`), charges (`MonthlyCharges`, `TotalCharges`), and target (`Churn`).

## Methodology

1. Download/cache and inspect the dataset.
2. Strip column whitespace, remove duplicates, convert `TotalCharges` to numeric, and retain valid target labels.
3. Record target distribution and missing-value audit in `reports/metrics.json`.
4. Engineer `tenure_years` and `avg_monthly_charge` inside the saved pipeline.
5. Split data using a stratified 80/20 split with `random_state=42`.
6. Impute numeric values with the median and categorical values with the most frequent value; scale numeric features and one-hot encode categorical features.
7. Tune and compare Logistic Regression and Random Forest with 3-fold cross-validation.
8. Evaluate accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
9. Select by highest holdout ROC-AUC, then F1, and save the complete preprocessing-plus-model pipeline with Joblib.

## Project structure

```text
customer-churn-ml/
├── data/                         # cached public CSV (ignored by git)
├── notebooks/churn_analysis.ipynb
├── src/preprocessing.py          # loading, cleaning, features, reusable pipeline
├── src/train.py                 # training, comparison, evaluation, serialization
├── src/predict.py               # inference using the saved artifact
├── app/                         # FastAPI backend (Phase 2)
│   ├── main.py                  # /health and /predict
│   └── schemas.py               # exact model input/output validation
├── tests/test_api.py
├── frontend/streamlit_app.py      # Streamlit HTTP client (Phase 3)
├── model/churn_pipeline.joblib
├── reports/                      # metrics, comparison CSV, confusion matrices
├── requirements.txt
├── README.md
└── .gitignore
```

## Run

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/predict.py
uvicorn app.main:app --reload
# In a second terminal:
streamlit run frontend/streamlit_app.py
```

The training command creates the cached dataset, evaluation reports, confusion-matrix images, and `model/churn_pipeline.joblib`. The artifact contains feature engineering, imputation, encoding, scaling, and the classifier, so inference does not refit preprocessing.

## Phase 2 API

Start from the project root with `uvicorn app.main:app --reload`. The API loads the existing artifact only; it never retrains it. Visit `http://127.0.0.1:8000/docs` for the interactive OpenAPI page.

- `GET /health` returns `{"status": "healthy"}`.
- `POST /predict` accepts the 20 raw model columns: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`.
- Risk labels are transparent business bands: probability `>= 0.70` is `High`; `>= 0.40` and `< 0.70` is `Medium`; `< 0.40` is `Low`. This does not change the model’s classification threshold.

## Phase 3 Streamlit frontend

The Streamlit dashboard lives in `frontend/streamlit_app.py` and communicates with FastAPI over HTTP. It never imports the API application, loads the Joblib file, preprocesses features, or predicts locally. Set `FASTAPI_URL` when the backend is not at the local default:

```bash
export FASTAPI_URL=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

The frontend is available at `http://localhost:8501`; FastAPI is available at `http://127.0.0.1:8000`. A typical workflow is: start FastAPI, open the dashboard, confirm the connected status, enter customer/service/account details, click **Predict Churn**, and review the API-provided prediction, probability, and risk level. Backend errors and validation failures are shown as friendly messages.

## Results

The exact scores are generated at runtime and written to `reports/metrics.json` and `reports/model_comparison.csv`. This avoids claiming fixed results when the source data or library versions change. The selected model is also recorded there.

## Future improvements

- Add probability calibration and a business-cost threshold instead of a default 0.5 cutoff.
- Add cross-validation confidence intervals and model explainability with feature importance/SHAP.
- Monitor drift and retrain on newly labeled customer outcomes during the later deployment stage.