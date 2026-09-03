# Customer Churn AI Deployment 🚀

An end-to-end, production-ready Machine Learning web application designed to predict telecom customer churn using advanced Scikit-learn pipelines, FastAPI endpoints, a modern React web dashboard, and a Streamlit analytics interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [📦 Deliverables & Artifacts](#-deliverables--artifacts)
- [📁 Repository Structure](#-repository-structure)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Setup ML Engine & FastAPI Server](#1-setup-ml-engine--fastapi-server)
  - [2. Setup React Web Frontend](#2-setup-react-web-frontend)
  - [3. Setup Backend Service (Optional)](#3-setup-backend-service-optional)
  - [4. Docker Deployment](#4-docker-deployment)
- [📡 API Documentation](#-api-documentation)
- [📊 Model Training & Evaluation](#-model-training--evaluation)
- [🧪 Testing](#-testing)
- [📄 License](#-license)

---

## 🎯 Overview

Customer retention is vital for subscription and telecom businesses. Missing a high-churn customer can result in lost recurring revenue, while over-targeting low-risk customers wastes retention budgets. 

This project provides an AI-driven solution that:
1. Trains and evaluates machine learning models (Logistic Regression vs. Random Forest) on the **IBM Telco Customer Churn** dataset.
2. Serializes an end-to-end preprocessing + prediction pipeline (`churn_pipeline.joblib`).
3. Exposes real-time prediction endpoints via a lightweight **FastAPI** service.
4. Delivers an interactive **React Dashboard** (with Tailwind CSS & Radix UI) and an analytical **Streamlit App** for customer success agents.

---

## 🏗️ System Architecture

```text
               +---------------------------------------------------+
               |               React Frontend (Port 3000)          |
               |       (Interactive Web UI for Agents & Staff)     |
               +-------------------------+-------------------------+
                                         |
                                         | HTTP REST Requests
                                         v
               +---------------------------------------------------+
               |             FastAPI ML Backend (Port 8000)        |
               |        (Implements /health and /predict APIs)     |
               +-------------------------+-------------------------+
                                         |
                                         | Loads Pipeline
                                         v
               +---------------------------------------------------+
               |         Scikit-Learn ML Pipeline (.joblib)        |
               |   (Imputers, Scalers, Encoders & Random Forest)  |
               +---------------------------------------------------+
                                         ^
                                         | Trained from
               +---------------------------------------------------+
               |             IBM Telco Churn Dataset               |
               +---------------------------------------------------+
```

---

## ✨ Key Features

- **End-to-End ML Pipeline**: Automated dataset loading, missing-value imputation, feature engineering (`tenure_years`, `avg_monthly_charge`), one-hot encoding, feature scaling, and classifier inference without refitting at request time.
- **Real-Time Risk Categorization**:
  - 🚨 **High Risk**: Churn Probability $\ge 70\%$
  - ⚠️ **Medium Risk**: Churn Probability $\ge 40\%$ and $< 70\%$
  - ✅ **Low Risk**: Churn Probability $< 40\%$
- **Modern React Dashboard**: Built with React 18, Tailwind CSS, Radix UI components, and real-time backend connection status monitors.
- **Streamlit Analytics Tool**: Lightweight internal Python dashboard for rapid data analysis and single-customer churn scoring.
- **Containerized & Deployable**: Pre-configured Dockerfile & `docker-compose.yml` setup for seamless deployment.

---

## 📦 Deliverables & Artifacts

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **ML Pipeline Artifact** | `customer-churn-ml/model/churn_pipeline.joblib` | Serialized Scikit-learn pipeline (preprocessing + feature engineering + trained Random Forest model). |
| **FastAPI ML Microservice** | `customer-churn-ml/app/` | Production REST API serving real-time predictions via `/predict` and `/health` endpoints. |
| **React Web App** | `frontend/` | Full-stack React dashboard with Tailwind CSS & Radix UI for interactive customer risk assessment. |
| **Streamlit Analytics App** | `customer-churn-ml/frontend/streamlit_app.py` | Standalone Python dashboard for internal data science & success team workflows. |
| **Backend Persistence API** | `backend/server.py` | Secondary FastAPI service with Async MongoDB Motor integration for status checks. |
| **Preprocessing & Training Modules** | `customer-churn-ml/src/` | Modular Python scripts (`preprocessing.py`, `train.py`, `predict.py`) for reproducible training workflows. |
| **EDA Notebook** | `customer-churn-ml/notebooks/churn_analysis.ipynb` | Exploratory data analysis notebook documenting dataset exploration & feature correlations. |
| **Evaluation Metrics & Reports** | `customer-churn-ml/reports/` | Detailed evaluation outputs (`metrics.json`, `model_comparison.csv`, confusion matrix visual artifacts). |
| **Docker & Compose Configs** | `customer-churn-ml/Dockerfile`, `docker-compose.yml` | Containerization manifests for container deployment. |
| **Regression & API Test Suites** | `tests/`, `customer-churn-ml/tests/` | Pytest suites verifying pipeline reload, preprocessing data cleaning, and REST API responses. |

---

## 📁 Repository Structure

```text
customer-churn-ai-deployment/
├── backend/                             # Secondary FastAPI & MongoDB status service
│   ├── server.py                        # FastAPI application with status routes & Motor MongoDB client
│   ├── requirements.txt                 # Backend Python dependencies (FastAPI, Motor, Pydantic, etc.)
│   ├── pytest.ini                       # Backend test configuration
│   └── .env                             # Backend environment variables
│
├── customer-churn-ml/                   # Core Machine Learning Engine & Serving API
│   ├── app/                             # FastAPI serving application
│   │   ├── main.py                      # REST endpoints (/health, /predict) with CORS configuration
│   │   └── schemas.py                   # Pydantic schemas for input features & prediction outputs
│   ├── data/                            # Public IBM Telco Customer Churn dataset cache (auto-downloaded)
│   ├── frontend/                        # Streamlit HTTP dashboard client
│   │   └── streamlit_app.py             # Streamlit application connecting to FastAPI backend
│   ├── model/                           # Serialized machine learning models
│   │   └── churn_pipeline.joblib        # Complete Scikit-learn preprocessing & classification pipeline
│   ├── notebooks/                       # Exploratory Data Analysis
│   │   └── churn_analysis.ipynb         # EDA notebook analyzing feature distributions & churn rates
│   ├── reports/                         # Training evaluation outputs & metrics
│   │   ├── metrics.json                 # JSON summary of model accuracy, precision, recall, F1, & ROC-AUC
│   │   └── model_comparison.csv         # Comparative metrics table across evaluated algorithms
│   ├── src/                             # Core ML pipeline modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py             # Data loading, cleaning, feature engineering & column transformers
│   │   ├── train.py                     # Pipeline training, cross-validation, evaluation & artifact saving
│   │   └── predict.py                   # CLI inference utility for saved pipeline
│   ├── tests/                           # ML API unit & integration tests
│   │   ├── __init__.py
│   │   └── test_api.py                  # Pytest suite for FastAPI endpoints
│   ├── Dockerfile                       # Docker image configuration for ML service
│   ├── docker-compose.yml               # Multi-container orchestration specification
│   ├── requirements.txt                 # ML engine Python dependencies
│   └── README.md                        # Dedicated ML module documentation
│
├── frontend/                            # Primary React Web Dashboard
│   ├── public/                          # Public static assets & index.html
│   ├── src/                             # React application source code
│   │   ├── components/                  # Reusable Radix UI & custom UI components
│   │   ├── App.js                       # Main dashboard component with prediction forms & status monitors
│   │   ├── App.css                      # Application styling and theme definitions
│   │   ├── index.js                     # React DOM root entrypoint
│   │   └── index.css                    # Tailwind CSS imports & base styles
│   ├── package.json                     # Node.js package definition & dependencies
│   ├── tailwind.config.js               # Tailwind CSS theme configuration
│   ├── craco.config.js                  # CRACO override configuration
│   └── jsconfig.json                    # JavaScript path mapping configuration
│
├── tests/                               # End-to-end system integration tests
│   ├── __init__.py
│   └── test_customer_churn_workflow.py  # System-wide regression tests verifying pipeline loading & data cleaning
│
├── test_reports/                        # Automated testing logs & execution reports
│   ├── iteration_1.json                 # Execution report logs
│   ├── iteration_2.json
│   ├── iteration_3.json
│   ├── iteration_4.json
│   └── pytest_churn_results.xml         # JUnit XML test results summary
│
├── .customer-churn-ai/                 # Project environment & platform configuration metadata
│   └── customer-churn-ai.yml            # Environment image and deployment configuration
├── test_result.md                       # Protocol documentation & testing tracking history
├── README.md                            # Main project documentation
└── package-lock.json                    # Workspace npm lockfile
```

---

## 🛠️ Tech Stack

- **Machine Learning & Backend**: Python 3.10+, Scikit-Learn, Pandas, NumPy, Joblib, FastAPI, Pydantic, Uvicorn, Motor, MongoDB.
- **Frontend Dashboard**: React 18, Tailwind CSS, Radix UI, Axios, Lucide React, Streamlit.
- **DevOps & Testing**: Docker, Docker Compose, Pytest.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:
- **Python**: version 3.10 or higher
- **Node.js**: version 18 or higher (with `npm` or `yarn`)
- **Git**

---

### 1. Setup ML Engine & FastAPI Server

1. Navigate to the `customer-churn-ml` directory:
   ```bash
   cd customer-churn-ml
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the ML model (downloads dataset, evaluates models, saves `churn_pipeline.joblib`):
   ```bash
   python src/train.py
   ```

5. Launch the FastAPI ML server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   *The ML API will be accessible at `http://127.0.0.1:8000` with documentation at `http://127.0.0.1:8000/docs`.*

6. *(Optional)* Launch the Streamlit dashboard in a separate terminal:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

---

### 2. Setup React Web Frontend

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables (optional, defaults to local or deployed API):
   Create a `.env` file in the `frontend/` directory:
   ```env
   REACT_APP_BACKEND_URL=http://127.0.0.1:8000
   ```

4. Start the React development server:
   ```bash
   npm start
   ```
   *The React web app will open at `http://localhost:3000`.*

---

### 3. Setup Backend Service (Optional)

If using the secondary status tracking backend with MongoDB:
1. Navigate to `backend/`:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables in `backend/.env`:
   ```env
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=churn_db
   ```
4. Start the backend server:
   ```bash
   uvicorn server:app --reload --port 8001
   ```

---

### 4. Docker Deployment

To build and run the ML API container using Docker:

```bash
cd customer-churn-ml
docker build -t customer-churn-api .
docker run -p 8000:8000 customer-churn-api
```

Or using Docker Compose:
```bash
cd customer-churn-ml
docker-compose up --build
```

---

## 📡 API Documentation

### System Health
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### Predict Churn
- **URL**: `/predict`
- **Method**: `POST`
- **Header**: `Content-Type: application/json`
- **Sample Request Payload**:
  ```json
  {
    "customerID": "CUST001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }
  ```
- **Sample Response**:
  ```json
  {
    "prediction": "Yes",
    "churn_probability": 0.7452,
    "risk_level": "High"
  }
  ```

### Batch Predict (CSV File Upload)
- **URL**: `/predict/file`
- **Method**: `POST`
- **Header**: `Content-Type: multipart/form-data`
- **Form Data**: `file` (CSV file with customer data)
- **Sample Response**:
  ```json
  [
    {
      "prediction": "Yes",
      "churn_probability": 0.6467,
      "risk_level": "Medium"
    }
  ]
  ```

---

## 📊 Model Training & Evaluation

The ML pipeline evaluates Logistic Regression and Random Forest models using 3-fold cross-validation. Selection prioritizes **ROC-AUC** followed by **F1 Score** to balance precision and recall.

Metrics and charts generated after running `python src/train.py`:
- `customer-churn-ml/reports/metrics.json`
- `customer-churn-ml/reports/model_comparison.csv`
- `customer-churn-ml/reports/logistic_regression_confusion_matrix.png`
- `customer-churn-ml/reports/random_forest_confusion_matrix.png`

---

## 🧪 Testing

Run system workflow regression tests:

```bash
# Run pytest across the workspace
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - feel free to customize and extend for your own applications.
