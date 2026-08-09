import os

base_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
judge_path = os.path.join(base_path, "unlabeled_judge_data_bundle", "unlabeled_judge_data")

items = os.listdir(judge_path)
print(f"Total items: {len(items)}")
print("First 10:", items[:10])