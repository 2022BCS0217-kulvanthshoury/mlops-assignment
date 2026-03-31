# src/preprocess.py

from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing(as_frame=True)
df = data.frame

# FULL dataset (v2)
df.to_csv("data/data_v2.csv", index=False)

# SMALL dataset (v1)
df.sample(500).to_csv("data/data_v1.csv", index=False)

print("Datasets created")