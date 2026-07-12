from fastapi import FastAPI
from app.schemas import HeartDiseaseRequest
from app.predictor import predict_heart_disease

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running."
    }


@app.post("/predict")
def predict(request: HeartDiseaseRequest):
    logger.info(f"Prediction requested: {request.model_dump()}")
    prediction, confidence = predict_heart_disease(
        request.model_dump()
    )

    logger.info(f"Prediction={prediction}, Confidence={confidence:.4f}")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }