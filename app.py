import pickle
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Load Model + Metadata
# -----------------------

model = tf.keras.models.load_model("house_price_model.keras")

with open("house_price_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

X_scaler = metadata["X_scaler"]
y_scaler = metadata["y_scaler"]

# -----------------------
# FastAPI App
# -----------------------

app = FastAPI(title="House Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HouseInput(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float
    house_age: float
    distance_from_city: float


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.post("/predict")
def predict_price(data: HouseInput):
    input_data = np.array([
        [
            data.area,
            data.bedrooms,
            data.bathrooms,
            data.house_age,
            data.distance_from_city
        ]
    ], dtype=float)

    input_scaled = X_scaler.transform(input_data)

    prediction_scaled = model.predict(input_scaled, verbose=0)

    prediction = y_scaler.inverse_transform(prediction_scaled)

    predicted_price = prediction[0][0]

    return {
        "area": data.area,
        "bedrooms": data.bedrooms,
        "bathrooms": data.bathrooms,
        "house_age": data.house_age,
        "distance_from_city": data.distance_from_city,
        "predicted_price": round(float(predicted_price), 2)
    }