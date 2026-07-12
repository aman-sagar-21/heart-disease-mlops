import joblib
import pandas as pd

MODEL_PATH = "models/heart_disease_rf_pipeline_v1.pkl"

model = joblib.load(MODEL_PATH)

def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0].max()

    return prediction, probability