import os

base_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
val_path = os.path.join(base_path, "validation_labeled_bundle")

for item in os.listdir(val_path):
    print(item)