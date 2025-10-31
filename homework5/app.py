from fastapi import FastAPI
import pickle

app = FastAPI()

with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Haloo! Model API is running!"}

@app.post("/predict")
def predict(client: dict):
    prob = model.predict_proba([client])[0, 1]
    return {"probability": prob}