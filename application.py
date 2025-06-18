from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from pathlib import Path


# load the model from disk
thisfile = Path(__file__).parent
modelfile = (thisfile / "random_forest_model.pkl").resolve()


# Joblib is an open-source library for the Python programming language that facilitates parallel processing, result caching and task distribution.
model = joblib.load(modelfile)

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

# api_url = "https://jsonplaceholder.typicode.com/todos/1"
# response = requests.get(api_url)
# response.json()