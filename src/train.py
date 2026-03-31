# src/train.py

import mlflow
import pandas as pd
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

NAME = "SHOURY"
ROLL = "BCS217"

mlflow.set_experiment(f"{ROLL}_experiment")

data_path = sys.argv[1]
use_feature_selection = len(sys.argv) > 2

df = pd.read_csv(data_path)

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Feature selection
if use_feature_selection:
    X = X[["MedInc", "AveRooms", "Population"]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

with mlflow.start_run():
    mlflow.log_param("data", data_path)
    mlflow.log_param("feature_selection", use_feature_selection)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl")