from datetime import datetime
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
import requests
from zoneinfo import ZoneInfo
from Random_forest import process_dataset, train_random_forest
from dotenv import load_dotenv
from dateutil.parser import parse

load_dotenv()

username = "Bitbybit11@login.nl"  # os.environ.get("API_USERNAME")
password1 = "Login123!!!"  # os.environ.get("API_PASSWORD1")

login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login"
data_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
login_data = {"email": username, "password": password1}

thisfile = Path(__file__).parent
modelfile = (thisfile / "random_forest_model.pkl").resolve()
model = joblib.load(modelfile)

app = FastAPI()

class DateInput(BaseModel):
    date: str  # Accepts ISO date/time strings

@app.get("/")
def read_root():
    return {"Hello": "World This api is up and running!"}

@app.post("/Predict/")
def predict(input: DateInput):
    # Robust date parsing
    try:
        dt = parse(input.date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

    features = [
        dt.month,
        dt.day,
        dt.weekday(),
        int(dt.month in [12, 1, 2]),  # is_winter
        int(dt.month in [6, 7, 8]),   # is_summer
    ]

    # Login to API
    try:
        login_response = requests.post(login_url, json=login_data, timeout=25)
        login_response.raise_for_status()
        token = login_response.json().get("accessToken")
        if not token:
            raise HTTPException(status_code=500, detail="No access token received from login.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login to API failed: {str(e)}")

    # Fetch dataset with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    try:
        dataset_response = requests.get(data_url, headers=headers, timeout=30)
        dataset_response.raise_for_status()
        dataset = dataset_response.json()
        if isinstance(dataset, dict) and "litter" in dataset:
            data_list = dataset["litter"]
        else:
            data_list = dataset
        X, y, confidences = process_dataset(data_list)
        model = train_random_forest(X, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetching dataset failed: {str(e)}")

    input_pred = model.predict([features])[0]
    all_preds = model.predict(X)
    same_pred_count = int(np.sum(all_preds == input_pred))

    return {
        "AdresPredection": input_pred,
        "CountOfPossibleLitter": same_pred_count,
        "MatchWithModel": round(same_pred_count / len(data_list) * 100, 2),
        "confidence": float(model.predict_proba([features]).max()),
        "timestamp": datetime.now(ZoneInfo("Europe/Amsterdam"))
    }