import { useEffect, useState } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL ||
  "https://customer-churn-ai-deployment.onrender.com";

const initialForm = {
  customerID: "CUST001",
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.0,
  TotalCharges: 840.0,
};

const Home = () => {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/health`);
        setBackendStatus(response.data.status);
      } catch (error) {
        setBackendStatus("Offline");
        console.error("Backend health check failed:", error);
      }
    };

    checkBackend();
  }, []);

  const handleChange = (event) => {
    const { name, value } = event.target;

    const numericFields = [
      "SeniorCitizen",
      "tenure",
      "MonthlyCharges",
      "TotalCharges",
    ];

    setForm((previous) => ({
      ...previous,
      [name]: numericFields.includes(name)
        ? name === "SeniorCitizen"
          ? Number(value)
          : Number(value)
        : value,
    }));
  };

  const predictChurn = async (event) => {
    event.preventDefault();

    setLoading(true);
    setResult(null);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/predict`,
        form,
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      setResult(response.data);
    } catch (error) {
      console.error("Prediction failed:", error);

      const message =
        error.response?.data?.detail ||
        "Prediction failed. Please check the backend.";

      setResult({
        error: message,
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm(initialForm);
    setResult(null);
  };

  return (
    <div className="churn-app">
      <div className="container">
        <header className="page-header">
          <div>
            <h1>Customer Churn Prediction</h1>
            <p>
              AI-powered customer churn prediction using a trained machine
              learning pipeline.
            </p>
          </div>

          <div className="backend-status">
            <span
              className={
                backendStatus === "healthy"
                  ? "status-dot healthy"
                  : "status-dot"
              }
            ></span>
            Backend: {backendStatus}
          </div>
        </header>

        <form onSubmit={predictChurn} className="prediction-form">
          <section className="form-section">
            <h2>Customer Information</h2>

            <div className="form-grid">
              <div className="form-group">
                <label>Customer ID</label>
                <input
                  type="text"
                  name="customerID"
                  value={form.customerID}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Gender</label>
                <select
                  name="gender"
                  value={form.gender}
                  onChange={handleChange}
                >
                  <option value="Female">Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>

              <div className="form-group">
                <label>Senior Citizen</label>
                <select
                  name="SeniorCitizen"
                  value={form.SeniorCitizen}
                  onChange={handleChange}
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Partner</label>
                <select
                  name="Partner"
                  value={form.Partner}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Dependents</label>
                <select
                  name="Dependents"
                  value={form.Dependents}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Tenure (months)</label>
                <input
                  type="number"
                  name="tenure"
                  min="0"
                  max="100"
                  value={form.tenure}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
          </section>

          <section className="form-section">
            <h2>Services</h2>

            <div className="form-grid">
              <div className="form-group">
                <label>Phone Service</label>
                <select
                  name="PhoneService"
                  value={form.PhoneService}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Multiple Lines</label>
                <select
                  name="MultipleLines"
                  value={form.MultipleLines}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </div>

              <div className="form-group">
                <label>Internet Service</label>
                <select
                  name="InternetService"
                  value={form.InternetService}
                  onChange={handleChange}
                >
                  <option value="DSL">DSL</option>
                  <option value="Fiber optic">Fiber optic</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Online Security</label>
                <select
                  name="OnlineSecurity"
                  value={form.OnlineSecurity}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Online Backup</label>
                <select
                  name="OnlineBackup"
                  value={form.OnlineBackup}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Device Protection</label>
                <select
                  name="DeviceProtection"
                  value={form.DeviceProtection}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Tech Support</label>
                <select
                  name="TechSupport"
                  value={form.TechSupport}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Streaming TV</label>
                <select
                  name="StreamingTV"
                  value={form.StreamingTV}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>

              <div className="form-group">
                <label>Streaming Movies</label>
                <select
                  name="StreamingMovies"
                  value={form.StreamingMovies}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                  <option value="No internet service">
                    No internet service
                  </option>
                </select>
              </div>
            </div>
          </section>

          <section className="form-section">
            <h2>Billing Information</h2>

            <div className="form-grid">
              <div className="form-group">
                <label>Contract</label>
                <select
                  name="Contract"
                  value={form.Contract}
                  onChange={handleChange}
                >
                  <option value="Month-to-month">Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>

              <div className="form-group">
                <label>Paperless Billing</label>
                <select
                  name="PaperlessBilling"
                  value={form.PaperlessBilling}
                  onChange={handleChange}
                >
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div className="form-group">
                <label>Payment Method</label>
                <select
                  name="PaymentMethod"
                  value={form.PaymentMethod}
                  onChange={handleChange}
                >
                  <option value="Bank transfer (automatic)">
                    Bank transfer (automatic)
                  </option>
                  <option value="Credit card (automatic)">
                    Credit card (automatic)
                  </option>
                  <option value="Electronic check">
                    Electronic check
                  </option>
                  <option value="Mailed check">Mailed check</option>
                </select>
              </div>

              <div className="form-group">
                <label>Monthly Charges</label>
                <input
                  type="number"
                  name="MonthlyCharges"
                  min="0"
                  step="0.01"
                  value={form.MonthlyCharges}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Total Charges</label>
                <input
                  type="number"
                  name="TotalCharges"
                  min="0"
                  step="0.01"
                  value={form.TotalCharges}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>
          </section>

          <div className="button-row">
            <button type="button" className="reset-button" onClick={resetForm}>
              Reset
            </button>

            <button type="submit" className="predict-button" disabled={loading}>
              {loading ? "Predicting..." : "Predict Churn"}
            </button>
          </div>
        </form>

        {result && (
          <section className="result-section">
            <h2>Prediction Result</h2>

            {result.error ? (
              <div className="error-box">
                <strong>Prediction failed</strong>
                <p>{result.error}</p>
              </div>
            ) : (
              <div className="result-card">
                <div className="result-item">
                  <span>Prediction</span>
                  <strong
                    className={
                      result.prediction === "Yes"
                        ? "danger-text"
                        : "safe-text"
                    }
                  >
                    {result.prediction === "Yes"
                      ? "Customer likely to churn"
                      : "Customer unlikely to churn"}
                  </strong>
                </div>

                <div className="result-item">
                  <span>Churn Probability</span>
                  <strong>
                    {(result.churn_probability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="result-item">
                  <span>Risk Level</span>
                  <strong>{result.risk_level}</strong>
                </div>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;