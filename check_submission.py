import pandas as pd

base_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
sub = pd.read_csv(base_path + r"\sample_submission.csv")

print("Shape:", sub.shape)
print("\nColumns:", sub.columns.tolist())
print("\nFirst 5 rows:\n", sub.head())
print("\nExample row_id values:\n", sub["row_id"].head(10).tolist())