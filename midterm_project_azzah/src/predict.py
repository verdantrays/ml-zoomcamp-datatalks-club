import os
import joblib
import numpy as np
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")

model = joblib.load(model_path)

@app.post("/predict")
def predict(data: dict):
    X = np.array([[data[k] for k in data]])
    pred = model.predict(X)[0]
    return {"prediction": float(pred)}

@app.get("/")
def root():
    return {"message": "Climate Model API is running, YEAYY"}