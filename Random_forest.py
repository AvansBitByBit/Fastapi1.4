import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from dateutil.parser import parse

def process_dataset(data_list):
    X, y, confidences = [], [], []
    for item in data_list:
        dt_str = item.get("time") or item.get("date")
        dt = parse(dt_str)
        temperature = item.get("temperature", 15.0)  # Default if missing
        features = [
            dt.month,
            dt.day,
            dt.weekday(),
            int(dt.month in [12, 1, 2]),  # is_winter
            int(dt.month in [6, 7, 8]),   # is_summer
            temperature
        ]
        X.append(features)
        y.append(item["location"])
        confidences.append(item.get("confidence", 1.0))
    return np.array(X), np.array(y), np.array(confidences)

def train_random_forest(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict_with_confidence(model, X, confidences):
    predictions = model.predict(X)
    results = []
    for pred, conf in zip(predictions, confidences):
        results.append({"prediction": pred, "confidence": conf})
    return results