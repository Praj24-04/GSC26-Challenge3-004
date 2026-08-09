import os

path = r'F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge'

# Check if the competition files are already fully downloaded and extracted
if os.path.isdir(path) and len(os.listdir(path)) > 0:
    print("Dataset already downloaded and extracted locally.")
    print("Path to competition files:", path)
else:
    # Configure kagglehub to use the F drive cache
    os.environ['KAGGLEHUB_CACHE'] = r'F:\IEEE\kaggle_cache'
    import kagglehub
    path = kagglehub.competition_download('fresco-early-job-failure-prediction-challenge')
    print("Path to competition files:", path)