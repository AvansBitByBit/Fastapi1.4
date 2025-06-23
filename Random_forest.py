from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def process_dataset(data_list):
    X = [[datetime.fromisoformat(item["time"]).timestamp()] for item in data_list]
    y = [item["trashType"] for item in data_list]
    return X, y

def train_random_forest(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model