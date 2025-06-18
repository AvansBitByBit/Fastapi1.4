import pandas as pd
from fastapi import FastAPI, requests
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import requests
from fastapi import Query

from roboflow import login


# Hierin staat de model die de fastapi krijg van onze webapi
class FutureFeatures(BaseModel):
    id: str
    time: str
    trashType: str
    location: str
    confidence: float
    celcius: float

# hierin laden we onze data vanuit de webapi
# eerst door de auth gaan
login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login"
login_data = {"username": "Bitbybit@login.nl", "password": "Login123!"}
login_response = requests.post(login_url, json=login_data)
token = login_response.json()["access_token"]

# daarna de data ophalen
headers = {"Authorization": f"Bearer {token}"}
api_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
dataJson = requests.get(api_url, headers=headers)
#parse de data
#ik krijg deze foutmelding niet weg.
# dataJson = data.json()
# RawData = pd.DataFrame(dataJson)


# load the model from disk
thisfile = Path(__file__).parent
modelfile = (thisfile / "random_forest_model.pkl").resolve()


# Joblib is an open-source library for the Python programming language that facilitates parallel processing, result caching and task distribution.
model = joblib.load(modelfile)

# initialize FastAPI
app = FastAPI()

class Features(BaseModel):
    features: List[float]

class TimeFrameRequest(BaseModel):
    days: int

# om de applicatie te testen
@app.get("/")
def read_root():
    return {"Python AI FastAPI model is live! goed bezig jongens": "Hello World!"}

@app.post("/predict/")
def predict(input: Features):
    prediction = model.predict(np.array([input.features]))
    print(prediction)
    return {"prediction": prediction.tolist()}


@app.post("/predictFuture")
def predict_future(input: FutureFeatures):

    hour = datetime.fromisoformat(input.time.replace("Z", "+00:00")).hour
    features = [input.confidence, input.celcius, hour]
    prediction = model.predict([features])
    return {"prediction": prediction.tolist()}

@app.post("/predict_trash_hotspots/")
def predict_trash_hotspots(request: TimeFrameRequest):
    # Authenticate
    login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login"
    login_data = {"email": "Bitbybit@login.nl", "password": "LoginAPI123!"}
    login_response = requests.post(login_url, json=login_data)
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Fetch litter data
    api_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
    data_response = requests.get(api_url, headers=headers)
    data = data_response.json()

    # Filter data by time frame
    cutoff = datetime.utcnow() - pd.Timedelta(days=request.days)
    filtered = [item for item in data if datetime.fromisoformat(item["time"].replace("Z", "+00:00")) >= cutoff]

    # Prepare features and locations
    predictions = []
    for item in filtered:
        hour = datetime.fromisoformat(item["time"].replace("Z", "+00:00")).hour
        features = [item["confidence"], item["celcius"], hour]
        pred = model.predict([features])[0]
        predictions.append({"location": item["location"], "prediction": pred})

    # Aggregate predictions by location
    location_counts = {}
    for p in predictions:
        loc = p["location"]
        location_counts[loc] = location_counts.get(loc, 0) + p["prediction"]

    # Find location(s) with highest predicted trash
    if location_counts:
        max_pred = max(location_counts.values())
        hotspots = [loc for loc, val in location_counts.items() if val == max_pred]
    else:
        hotspots = []

    return {"hotspots": hotspots, "location_predictions": location_counts}