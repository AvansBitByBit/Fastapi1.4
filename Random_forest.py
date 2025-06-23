from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def process_dataset(data_list):
    X = [item["temperature"] for item in data_list]
    y = [item["location"] for item in data_list]
    return X, y

def train_random_forest(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return models