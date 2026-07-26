import os

base_path = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
train_path = os.path.join(base_path, "training_data_bundle", "training_data")

total_files = 0
total_size_mb = 0

for month in os.listdir(train_path):
    month_dir = os.path.join(train_path, month)
    if not os.path.isdir(month_dir):
        print("Skipping non-folder item:", month)
        continue
    files = os.listdir(month_dir)
    total_files += len(files)
    for f in files:
        total_size_mb += os.path.getsize(os.path.join(month_dir, f)) / (1024 * 1024)

print("Total files across all months:", total_files)
print("Total size (MB):", round(total_size_mb, 2))
print("Total size (GB):", round(total_size_mb / 1024, 2))