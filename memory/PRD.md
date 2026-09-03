# Customer Churn ML Project

## Original problem statement
Build the model-building stage of a professional customer churn prediction project using a real public telecom dataset, with reproducible loading, inspection, cleaning, EDA, feature engineering, preprocessing, model comparison, tuning, evaluation, and Joblib serialization. Do not build deployment, API, frontend, authentication, database, Docker, or cloud components yet.

## Architecture decisions
- Isolated Python project under `/app/customer-churn-ml/`.
- IBM Telco Customer Churn CSV is downloaded from its documented public raw source and cached locally.
- A scikit-learn Pipeline owns feature engineering, imputation, encoding, scaling, and classification.
- Logistic Regression and Random Forest are tuned with stratified cross-validation and compared on a stratified holdout set.
- The selected pipeline is serialized as `model/churn_pipeline.joblib`; inference imports the package-qualified transformer for portability.

## Implemented
- Dataset loading/cache, duplicate audit, missing-value audit, numeric conversion, and target validation.
- Notebook EDA for target distribution, numerical distributions/grouped summaries, and categorical churn rates/plots.
- Feature engineering for tenure in years and average monthly charge.
- Tuned Logistic Regression and Random Forest with accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
- Metrics JSON, comparison CSV, confusion-matrix images, README, requirements, prediction script, and verified reload/inference flow.

## Prioritized backlog
- P0: None for the model-building scope.
- P1: Add calibrated probabilities and a business-cost threshold for retention outreach.
- P1: Add feature explainability and confidence intervals.
- P2: Add drift monitoring and scheduled retraining during the later deployment phase.

## Phase 2 backend implementation
- Added `customer-churn-ml/app/main.py`, `app/schemas.py`, and `app/__init__.py`.
- Added strict Pydantic validation for the exact 20 raw input columns and discovered categorical domains.
- Added `GET /health`, `POST /predict`, OpenAPI docs, cached artifact loading, error mapping, and transparent risk bands.
- Added six API tests covering health, real prediction, invalid/missing/domain input, model-load failure, and prediction failure.
- Verification: pytest 6 passed; live health/predict/docs passed; artifact SHA-256 remains `721ff5a23a1a7cea0b6d4c03bc1cc6e435df855625ef04c8c478cfe31e28feb3`.

## Phase 2 backlog
- P1: Add frontend only when the next project phase begins.
- P2: Add production deployment configuration only after frontend/API review.

## Phase 3 frontend implementation
- Added `customer-churn-ml/frontend/streamlit_app.py`, an HTTP-only Streamlit dashboard.
- The UI uses the exact 20 API fields, organized customer/service/account sections, `FASTAPI_URL`, health status, loading state, friendly errors, and API-driven prediction cards.
- Streamlit never imports FastAPI, Joblib, or ML preprocessing/prediction code.
- Verified connected flow at `http://127.0.0.1:8501`, unavailable-backend flow at `FASTAPI_URL=http://127.0.0.1:8999`, and real default result `85.09% / High`.
- Dependencies: Streamlit 1.41.1 and requests; backend regression remains 6/6 passed.

## Phase 3 backlog
- P1: Add optional explanation visuals only through a future backend response extension.
- P2: Add Docker/cloud deployment in a later phase.

## Artifact recovery for GitHub (2026-01)
- Issue: `/app/customer-churn-ml/.gitignore` excluded the immutable production artifact via `model/*.joblib`, blocking it from source control.
- Fix: Added narrow negation `!model/churn_pipeline.joblib` after the broad ignore rule. No model bytes touched. No backend code touched.
- Verification (iteration_4.json):
  - SHA-256 unchanged and equals `721ff5a23a1a7cea0b6d4c03bc1cc6e435df855625ef04c8c478cfe31e28feb3`.
  - `git check-ignore` exit code 1 (NOT ignored); last matching rule is the negation at line 7.
  - Backend `/health` and `/predict` end-to-end pass; pytest 6/6.
- Pending user step (blocked on platform policy): User must use Emergent "Save to GitHub" to commit/push the artifact to `customer-churn-ai-deployment` on `main`. Agent cannot perform Git write actions.
