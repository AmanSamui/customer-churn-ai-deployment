"""Streamlit UI that communicates with the FastAPI prediction service over HTTP only."""

import io
import os

import pandas as pd
import requests
import streamlit as st

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 30

YES_NO = ["Yes", "No"]
SERVICE_OPTIONS = ["Yes", "No", "No internet service"]


def backend_health() -> bool:
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=3)
        return response.status_code == 200 and response.json().get("status") == "healthy"
    except (requests.RequestException, ValueError):
        return False


def format_api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", "The backend rejected the request.")
        if isinstance(detail, list):
            return "Please review the highlighted input values and try again."
        return str(detail)
    except ValueError:
        return f"The backend returned HTTP {response.status_code}."


def render_result(result: dict) -> None:
    prediction = result["prediction"]
    probability = float(result["churn_probability"])
    risk = result["risk_level"]
    risk_class = risk.lower()
    st.markdown('<div class="result-kicker">MODEL ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-title">{"Likely to Churn" if prediction == "Yes" else "Likely to Stay"}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn probability", f"{probability:.2%}")
    with col2:
        st.markdown(f'<div class="risk-badge {risk_class}">{risk} risk</div>', unsafe_allow_html=True)
    st.progress(probability, text=f"Churn likelihood · {probability:.2%}")
    st.caption("Risk bands: Low < 40% · Medium 40–69.99% · High ≥ 70%")


st.set_page_config(page_title="Customer Churn Prediction", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#14211f; --muted:#62716e; --mint:#d9f2e4; --green:#1c6b52; --orange:#d97943; --line:#d8e4df; }
    .stApp { background: radial-gradient(circle at 85% 4%, #e9f8ef 0, transparent 31%), #f6faf7; color:var(--ink); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; }
    p, label, .stCaption { font-family:'DM Sans',sans-serif; }
    .hero { padding: 2.5rem 0 1.4rem; border-bottom:1px solid var(--line); margin-bottom:1.8rem; }
    .eyebrow, .result-kicker { color:var(--orange); letter-spacing:.16em; font-size:.72rem; font-weight:700; }
    .hero h1 { font-size:clamp(2.2rem,5vw,4.2rem); line-height:.98; margin:.45rem 0 .7rem; }
    .hero p { color:var(--muted); max-width:700px; font-size:1.05rem; }
    [data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:20px; padding:1.25rem 1.5rem 1.5rem; box-shadow:0 18px 45px rgba(35,83,63,.06); }
    .section-title { font-family:'Space Grotesk'; font-weight:700; font-size:1.15rem; margin:1rem 0 .2rem; }
    .section-copy { color:var(--muted); font-size:.84rem; margin-bottom:.65rem; }
    .result-panel { background:#14211f; border-radius:20px; padding:1.7rem; color:white; min-height:320px; box-shadow:0 18px 45px rgba(20,33,31,.16); }
    .result-panel .result-title { color:white; font-family:'Space Grotesk'; font-size:2rem; font-weight:700; margin:.4rem 0 1.3rem; }
    .risk-badge { display:inline-block; margin-top:.35rem; padding:.6rem 1rem; border-radius:999px; font-weight:700; text-align:center; }
    .risk-badge.high { color:#fff; background:#b94f3e; } .risk-badge.medium { color:#fff; background:#c77b34; } .risk-badge.low { color:#fff; background:#3c8b67; }
    .backend-ok { color:#2f805e; font-weight:600; } .backend-down { color:#b94f3e; font-weight:600; }
    div[data-testid="stMetricValue"] { color:white; }
    </style>
    """, unsafe_allow_html=True,
)

st.markdown('<div class="hero"><div class="eyebrow">TELCO · MACHINE LEARNING</div><h1>Customer Churn<br>Prediction</h1><p>AI-powered telecom customer churn risk prediction. Predict churn for individual customer profiles or perform batch predictions by uploading CSV files.</p></div>', unsafe_allow_html=True)

healthy = backend_health()
status_text = "● Backend connected" if healthy else "● Backend unavailable"
status_class = "backend-ok" if healthy else "backend-down"
st.markdown(f'<div class="{status_class}">{status_text} · {FASTAPI_URL}</div>', unsafe_allow_html=True)
st.write("")

tab_single, tab_batch = st.tabs(["👤 Single Customer Prediction", "📁 Batch CSV File Upload"])

with tab_single:
    form_col, result_col = st.columns([1.35, .8], gap="large")
    with form_col:
        with st.form("customer_form"):
            st.markdown('<div class="section-title">Customer information</div><div class="section-copy">Start with the customer profile and account relationship.</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                customer_id = st.text_input("Customer ID", value="DEMO-0001")
                gender = st.selectbox("Gender", ["Female", "Male"])
            with c2:
                senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
                partner = st.selectbox("Partner", YES_NO)
            with c3:
                dependents = st.selectbox("Dependents", YES_NO)
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=2, step=1)

            st.markdown('<div class="section-title">Service information</div><div class="section-copy">Choose the services currently attached to this account.</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                phone_service = st.selectbox("Phone service", YES_NO)
                multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
                internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online security", SERVICE_OPTIONS)
            with c2:
                online_backup = st.selectbox("Online backup", SERVICE_OPTIONS)
                device_protection = st.selectbox("Device protection", SERVICE_OPTIONS)
                tech_support = st.selectbox("Tech support", SERVICE_OPTIONS)
                streaming_tv = st.selectbox("Streaming TV", SERVICE_OPTIONS)
            with c3:
                streaming_movies = st.selectbox("Streaming movies", SERVICE_OPTIONS)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless billing", YES_NO)
                payment_method = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

            st.markdown('<div class="section-title">Account information</div><div class="section-copy">Use the customer’s current monthly and lifetime charges.</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=53.85, step=1.0, format="%.2f")
            with c2:
                total_charges = st.number_input("Total charges", min_value=0.0, value=108.15, step=1.0, format="%.2f")
            submitted = st.form_submit_button("Predict Churn", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "customerID": customer_id.strip(), "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }
        with result_col:
            with st.container(border=True):
                if not payload["customerID"]:
                    st.error("Please enter a customer ID before predicting.")
                else:
                    with st.spinner("Analyzing customer..."):
                        try:
                            response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT)
                            if response.status_code == 200:
                                result = response.json()
                                if not {"prediction", "churn_probability", "risk_level"}.issubset(result):
                                    st.error("The backend returned an unexpected response.")
                                else:
                                    render_result(result)
                            elif response.status_code == 422:
                                st.error(f"Please check the submitted values. {format_api_error(response)}")
                            elif response.status_code >= 500:
                                st.error("The prediction service encountered an error. Please try again shortly.")
                            else:
                                st.error(format_api_error(response))
                        except requests.Timeout:
                            st.error("The backend took too long to respond. Please try again.")
                        except requests.ConnectionError:
                            st.error("The prediction backend is unavailable. Start FastAPI and try again.")
                        except requests.RequestException:
                            st.error("Could not reach the prediction backend.")
                        except ValueError:
                            st.error("The backend returned an unreadable response.")
    elif not healthy:
        with result_col:
            st.warning("Start FastAPI to enable predictions. The form will connect automatically once the backend is available.")

with tab_batch:
    st.markdown('<div class="section-title">Batch Customer Churn Analysis</div><div class="section-copy">Upload a CSV file containing multiple customer records to generate batch predictions.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(df_preview)} rows** from `{uploaded_file.name}`")
            st.dataframe(df_preview.head(5), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Running predictions across dataset..."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{FASTAPI_URL}/predict/file", files=files, timeout=REQUEST_TIMEOUT)

                    if response.status_code == 200:
                        results = response.json()
                        res_df = pd.DataFrame(results)
                        output_df = pd.concat([df_preview.reset_index(drop=True), res_df], axis=1)

                        st.success(f"Successfully processed {len(output_df)} records!")

                        m1, m2, m3 = st.columns(3)
                        high_risk_count = (output_df["risk_level"] == "High").sum()
                        churn_count = (output_df["prediction"] == "Yes").sum()
                        avg_prob = output_df["churn_probability"].mean()

                        m1.metric("Predicted Churners", f"{churn_count} / {len(output_df)}")
                        m2.metric("High Risk Accounts", f"{high_risk_count}")
                        m3.metric("Average Churn Probability", f"{avg_prob:.2%}")

                        st.dataframe(output_df, use_container_width=True)

                        csv_bytes = output_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Batch Results (CSV)",
                            data=csv_bytes,
                            file_name=f"churn_predictions_{uploaded_file.name}",
                            mime="text/csv",
                            type="secondary"
                        )
                    else:
                        st.error(f"Batch prediction failed: {format_api_error(response)}")
        except Exception as err:
            st.error(f"Error parsing uploaded file: {err}")