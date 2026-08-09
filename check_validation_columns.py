import pandas as pd
import os

val_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge\validation_labeled_bundle"
files = [f for f in os.listdir(val_path) if f.endswith(".parquet")]

df = pd.read_parquet(os.path.join(val_path, files[0]))
print("Columns:", df.columns.tolist())
print("\nSample:\n", df.head(3))