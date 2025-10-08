# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("safety_model.pkl")

# Define API
app = FastAPI(
    title="AI Safety Model API",
    description="Detects abusive, escalating, crisis, and age-inappropriate messages.",
    version="2.0"
)

class Message(BaseModel):
    text: str

# Label names (order must match training labels)
LABELS = ['abuse', 'escalation', 'crisis', 'age_inappropriate']

@app.get("/")
def root():
    return {"message": "AI Safety API is running."}

@app.post("/predict")
def predict(message: Message):
    prediction = model.predict([message.text])[0]
    results = dict(zip(LABELS, map(int, prediction)))
    return {
        "input": message.text,
        "safety_flags": results
    }
