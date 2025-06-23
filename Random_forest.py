import numpy as np
from sklearn.ensemble import RandomForestClassifier

def process_dataset(data_list):
    # Extract features, targets, and confidence values
    X = [[item["temperature"]] for item in data_list]
    y = [item["location"] for item in data_list]
    confidences = [item["confidence"] for item in data_list]
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