import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
import requests
from Random_forest import process_dataset, train_random_forest
from dotenv import load_dotenv
load_dotenv()


username = os.environ.get("API_USERNAME")
password1 = os.environ.get("API_PASSWORD1")

# if not username or not password1:
#     raise RuntimeError("API credentials not configured. Please set API_USERNAME and API_PASSWORD1.")

login_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/account/login"
data_url = "https://bitbybit-api.orangecliff-c30465b7.northeurope.azurecontainerapps.io/litter"
login_data = {"email": username, "password": password1}

thisfile = Path(__file__).parent
modelfile = (thisfile / "random_forest_model.pkl").resolve()
model = joblib.load(modelfile)

app = FastAPI()

class Features(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"Hello": "World This api is up and running!"}


@app.post("/Predict/")
def predict(input: Features):
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
    # login reponse worked
    try:
        dataset_response = requests.get(data_url, headers=headers, timeout=30)
        dataset_response.raise_for_status()
        dataset = dataset_response.json()
        print("Fetched dataset:", dataset)  # Debug: check structure

        if isinstance(dataset, dict) and "litter" in dataset:
            data_list = dataset["litter"]
        else:
            data_list = dataset
        X, y = process_dataset(data_list)
        model = train_random_forest(X, y)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fetching dataset failed: {str(e)}")

    prediction = model.predict(np.array([input.features]))
    return {"prediction": prediction.tolist()}