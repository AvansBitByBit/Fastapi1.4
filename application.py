from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path
<<<<<<< Updated upstream

=======
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import requests
>>>>>>> Stashed changes

# load the model from disk
thisfile = Path(__file__).parent
modelfile = (thisfile / "random_forest_model.pkl").resolve()


# Joblib is an open-source library for the Python programming language that facilitates parallel processing, result caching and task distribution.
model = joblib.load(modelfile)


# make a request to the API to ensure it's running
# api_url = "https://bitbybit-api--0000005.orangecliff-c30465b7.northeurope.azurecontainerapps.io/api/dummy"
# response = requests.get(api_url)

# initialize FastAPI
app = FastAPI()

class Features(BaseModel):
    features: List[float]

# om de applicatie te testen
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict(input: Features):
    prediction = model.predict(np.array([input.features]))
    print(prediction)
    return {"prediction": prediction.tolist()}

