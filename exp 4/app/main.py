from fastapi import FastAPI
import numpy as np
from app.models import IrisFeatures
from app.ml_model import ml_model

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML API running"}

@app.post("/predict")
def predict(features: IrisFeatures):

    data = np.array([[

        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width

    ]])

    result = ml_model.predict(data)

    return 
