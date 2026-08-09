import pandas as pd
import os

judge_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge\unlabeled_judge_data_bundle\unlabeled_judge_data"

sample_file = None
for month in os.listdir(judge_path):
    month_dir = os.path.join(judge_path, month)
    if os.path.isdir(month_dir):
        files = [f for f in os.listdir(month_dir) if f.endswith(".parquet")]
        if files:
            sample_file = os.path.join(month_dir, files[0])
            break

df = pd.read_parquet(sample_file)
print("Columns:", df.columns.tolist())
print("\nSample jid values:")
print(df["jid"].head(10).tolist())