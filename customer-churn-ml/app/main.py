"""FastAPI interface for the existing, immutable churn pipeline artifact."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

from app.schemas import CustomerFeatures, PredictionResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "churn_pipeline.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Prediction API backed by the verified IBM Telco churn pipeline.",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model():
    """Load the saved pipeline once; never train or alter it at request time."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    return load(MODEL_PATH)


def risk_level(probability: float) -> str:
    """Transparent business label: >=.70 High, >=.40 Medium, otherwise Low."""
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    try:
        pipeline = load_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {exc}"
        ) from exc

    try:
        frame = pd.DataFrame([customer.model_dump()])
        predicted = str(pipeline.predict(frame)[0])

        if not hasattr(pipeline, "predict_proba"):
            raise RuntimeError(
                "Loaded model does not support probability predictions"
            )

        probabilities = pipeline.predict_proba(frame)[0]
        classes = list(getattr(pipeline, "classes_", ["No", "Yes"]))
        yes_index = classes.index("Yes")
        probability = float(probabilities[yes_index])

        return PredictionResponse(
            prediction=predicted,
            churn_probability=round(probability, 4),
            risk_level=risk_level(probability),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc


@app.post("/predict/file", response_model=list[PredictionResponse], tags=["prediction"])
async def predict_file(file: UploadFile = File(...)) -> list[PredictionResponse]:
    """Accept a CSV file containing customer rows and return batch predictions."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        pipeline = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {exc}") from exc

    try:
        contents = await file.read()
        import io
        frame = pd.read_csv(io.BytesIO(contents))
        
        predictions = pipeline.predict(frame)
        probabilities = pipeline.predict_proba(frame)
        classes = list(getattr(pipeline, "classes_", ["No", "Yes"]))
        yes_index = classes.index("Yes")

        results = []
        for i in range(len(frame)):
            pred = str(predictions[i])
            prob = float(probabilities[i][yes_index])
            results.append(
                PredictionResponse(
                    prediction=pred,
                    churn_probability=round(prob, 4),
                    risk_level=risk_level(prob),
                )
            )
        return results
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process CSV file: {exc}") from exc