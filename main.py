from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Customer Churn Predictor")

class ChurnInput(BaseModel):
    seniorcitizen: int
    monthlycharges: float
    tenure: int

@app.post("/predict")
def predict_churn(data: ChurnInput):
    input_data = np.array([[data.seniorcitizen, data.monthlycharges, data.tenure]])
    prediction = model.predict(input_data)[0]
    return {"churn": int(prediction)}
