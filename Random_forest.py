import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dateutil.parser import parse

def process_dataset(data_list):
    X, y, confidences = [], [], []
    for item in data_list:
        dt_str = item.get("time") or item.get("date") # de post body kan time en date verwachten
        dt = parse(dt_str)
        temperature = item.get("temperature", 15.0)  # krijgt temperatuur mee, default 15.0 als niet aanwezig
        features = [ #features voor de Random Forest
            dt.month,
            dt.day,
            dt.weekday(),
            int(dt.month in [12, 1, 2]),
            int(dt.month in [6, 7, 8]),
            temperature
        ]
        X.append(features)
        y.append(item["location"]),#We willen locatie voorspellen
        confidences.append(item.get("confidence", 1.0)) # default confidence is 1.0 if not present
    return np.array(X), np.array(y), np.array(confidences) #returnt de gegevens als  arrays

def train_random_forest(X, y):
    model = RandomForestClassifier() # Initialize the Random Forest model
    model.fit(X, y) # Train the model with the features and labels
    return model

def predict_with_confidence(model, X, confidences):
    predictions = model.predict(X)
    results = []
    for pred, conf in zip(predictions, confidences):
        results.append({"prediction": pred, "confidence": conf})
    return results