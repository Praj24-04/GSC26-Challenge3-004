"""
train_baseline.py

GSC26 Challenge 3 - Data Center Job Failure Prediction
Team 004 (Prajakta Jadhav, Aariya Vora)

Loads a sample of raw hourly FRESCO Parquet files, aggregates them into
one row per job, builds the binary failure target, and trains a baseline
Random Forest classifier.

Intended to be run inside a Kaggle Notebook with the FRESCO competition
data attached as input (see README.md for dataset details).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------
# STEP 1: Load a manageable sample of raw hourly files
# (Using a few month folders for this baseline -- not all 46,795
# participant files. Scaling this up is listed as a next step.)
# ---------------------------------------------------------------
TRAIN_ROOT = (
    "/kaggle/input/competitions/fresco-early-job-failure-prediction-challenge"
    "/training_data_bundle/training_data"
)
SAMPLE_MONTHS = ["2013-10", "2014-03", "2014-05"]
FILES_PER_MONTH = 100  # small sample per month, for fast iteration


def load_sample(train_root, months, files_per_month):
    all_rows = []
    for month in months:
        month_dir = os.path.join(train_root, month)
        files = os.listdir(month_dir)[:files_per_month]
        for fname in files:
            fpath = os.path.join(month_dir, fname)
            df = pd.read_parquet(fpath)
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------
# STEP 2: Aggregate hourly snapshots into ONE row per job
# ---------------------------------------------------------------
def build_job_features(raw):
    job_features = raw.groupby("jid").agg(
        nhosts=("nhosts", "max"),
        ncores=("ncores", "max"),
        timelimit=("timelimit", "max"),
        cpu_mean=("value_cpuuser", "mean"),
        cpu_max=("value_cpuuser", "max"),
        mem_mean=("value_memused", "mean"),
        mem_max=("value_memused", "max"),
        nfs_mean=("value_nfs", "mean"),
        block_mean=("value_block", "mean"),
        num_snapshots=("time", "count"),
        exitcode=("exitcode", "first"),  # label source only, never a feature
    ).reset_index()
    return job_features


# ---------------------------------------------------------------
# STEP 3: Build the target label
# CANCELLED jobs are excluded per competition rules (not part of the
# scored positive or negative class).
# ---------------------------------------------------------------
def build_target(job_features):
    job_features = job_features[job_features["exitcode"] != "CANCELLED"].copy()
    job_features["target"] = job_features["exitcode"].apply(
        lambda x: 1 if x in ("FAILED", "TIMEOUT", "NODE_FAIL") else 0
    )
    return job_features


FEATURE_COLS = [
    "nhosts", "ncores", "timelimit", "cpu_mean", "cpu_max",
    "mem_mean", "mem_max", "nfs_mean", "block_mean", "num_snapshots",
]


# ---------------------------------------------------------------
# STEP 4: Train and evaluate a baseline classifier
# ---------------------------------------------------------------
def train_and_evaluate(job_features):
    X = job_features[FEATURE_COLS].fillna(0)
    y = job_features["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n=== Baseline model results ===")
    print(classification_report(
        y_test, preds, target_names=["Completed", "Failed/Timeout/NodeFail"]
    ))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    print("\nFeature importances:")
    for name, importance in sorted(
        zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {importance:.3f}")

    return clf


def main():
    print("Loading raw hourly Parquet sample...")
    raw = load_sample(TRAIN_ROOT, SAMPLE_MONTHS, FILES_PER_MONTH)
    print("Raw rows loaded:", raw.shape)

    print("\nAggregating into per-job features...")
    job_features = build_job_features(raw)
    print("Unique jobs after aggregation:", job_features.shape[0])
    print("\nExit code distribution:\n", job_features["exitcode"].value_counts())

    job_features = build_target(job_features)
    print("\nTarget distribution:\n", job_features["target"].value_counts())

    train_and_evaluate(job_features)


if __name__ == "__main__":
    main()
