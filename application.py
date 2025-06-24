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

username = "Bitbybit11@login.nl"#os.environ.get("API_USERNAME")  #secrets in azure
password1 = "Login123!!!"#os.environ.get("API_PASSWORD") #secrets in azure

login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login" #api login url
data_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter" #api data url
login_data = {"email": username, "password": password1} #login data voor de api

thisfile = Path(__file__).parent # Get the directory of the current file
modelfile = (thisfile / "random_forest_model.pkl").resolve() # Resolve the path to the model file
model = joblib.load(modelfile) # Load the pre-trained model

app = FastAPI()

class DateInput(BaseModel):
    date: str  # Accepts ISO date/time strings
    temperature: float # Accepts temperature as a float
@app.get("/") #start api also knowen as root
def read_root():
    return {"Hello": "Welkom bij de Litter Prediction API!"}

@app.post("/Predict/") #post request voor de predictie
def predict(input: DateInput):
    try:
        dt = parse(input.date) # Parse the date string into a datetime object
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

    features = [
        dt.month,
        dt.day,
        dt.weekday(),
        int(dt.month in [12, 1, 2]),  # is_winter
        int(dt.month in [6, 7, 8]),   # is_summer
        input.temperature
    ]

    # Login to API
    try:
        login_response = requests.post(login_url, json=login_data, timeout=25) # Post request to login
        login_response.raise_for_status() # Raise an error for bad responses
        token = login_response.json().get("accessToken") # Get the access token from the response
        if not token:
            raise HTTPException(status_code=500, detail="No access token received from login.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login to API failed: {str(e)}")

    # Fetch dataset with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    try:
        dataset_response = requests.get(data_url, headers=headers, timeout=30) # Get request to fetch the dataset
        dataset_response.raise_for_status()  # Raise an error for bad responses
        dataset = dataset_response.json()  # Parse the JSON response
        if isinstance(dataset, dict) and "litter" in dataset:  # Check if the dataset is a dictionary with a "litter" key
            data_list = dataset["litter"]  # Extract the list of litter data
        else:
            data_list = dataset  # If the dataset is already a list, use it directly
        X, y, confidences = process_dataset(data_list)  # Process the dataset to extract features and labels
        model = train_random_forest(X, y)  # Train the Random Forest model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetching dataset failed: {str(e)}")

    input_pred = model.predict([features])[0]  # Predict the location using the input features
    all_preds = model.predict(X)  # Predict all locations in the dataset
    same_pred_count = int(np.sum(all_preds == input_pred)) # Count how many times the predicted location appears in the dataset

    return {
        "AdresPredection": input_pred, # Predicted location based on the input features
        "CountOfPossibleLitter": same_pred_count, # Count of how many times the predicted location appears in the dataset
        "MatchWithModel": round(same_pred_count / len(data_list) * 100, 2), # Percentage match with the model
        "confidence": float(model.predict_proba([features]).max()), # Confidence of the prediction
        "timestamp": datetime.now(ZoneInfo("Europe/Amsterdam")) # Current timestamp in Amsterdam timezone
    }