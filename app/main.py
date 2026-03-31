from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

NAME = "SHOURY"
ROLL = "BCS217"

class InputData(BaseModel):
    data: List[float]

@app.get("/health")
def health():
    return {"name": NAME, "roll": ROLL}

@app.post("/predict")
def predict(input: InputData):
    pred = model.predict([input.data]).tolist()
    return {
        "prediction": pred,
        "name": NAME,
        "roll": ROLL
    }