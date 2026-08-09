# -*- coding: utf-8 -*-
"""
generate_submission.py

Builds the final Kaggle submission for the FRESCO Early Job Failure
Prediction Challenge.

Steps:
1. Stream through the judge dataset (one file at a time, memory-safe --
   same approach as training) and build EARLY-WINDOW per-job features,
   identical in structure to what the models were trained on.
2. Load the 3 saved per-cluster models (model_S, model_C, model_Anvil).
3. Route each job to its correct cluster's model and predict a failure
   probability.
4. Merge onto the OFFICIAL sample_submission.csv row_id list (the
   authoritative source of required rows) -- any row we couldn't build
   features for gets a sensible default probability, clearly reported.
5. Write submission.csv and zip it, containing ONLY that one file, per
   the submission contract.
"""

import gc
import os
import zipfile
import joblib
import numpy as np
import pandas as pd

BASE_PATH = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
JUDGE_ROOT = os.path.join(BASE_PATH, "unlabeled_judge_data_bundle", "unlabeled_judge_data")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

MODEL_DIR = "models"
EARLY_WINDOW_SNAPSHOTS = 2  # MUST match what train_models.py used

NEEDED_COLUMNS = [
    "time", "submit_time", "start_time", "jid",
    "nhosts", "ncores", "timelimit",
    "value_cpuuser", "value_memused", "value_nfs", "value_block",
]

FEATURE_COLUMNS = [
    "nhosts", "ncores", "timelimit", "cpu_mean", "cpu_max",
    "mem_mean", "mem_max", "nfs_mean", "block_mean",
    "num_early_snapshots", "queue_wait_seconds",
]


def cluster_code_from_filename(filename):
    if filename.endswith("_S.parquet"):
        return "S"
    elif filename.endswith("_C.parquet"):
        return "C"
    else:
        return "Anvil"


def to_naive(ts):
    if ts is None or pd.isna(ts):
        return ts
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None) if hasattr(ts, "tz_localize") else ts.replace(tzinfo=None)
    return ts


def new_accumulator_entry():
    return {
        "n": 0,
        "cpu_sum": 0.0, "cpu_max": -np.inf,
        "mem_sum": 0.0, "mem_max": -np.inf,
        "nfs_sum": 0.0, "block_sum": 0.0,
        "nhosts": None, "ncores": None, "timelimit": None,
        "submit_time": None, "start_time": None,
    }


def update_entry_with_hour(entry, hour_row):
    entry["n"] += 1
    entry["cpu_sum"] += hour_row["value_cpuuser"] if pd.notna(hour_row["value_cpuuser"]) else 0.0
    entry["cpu_max"] = max(entry["cpu_max"], hour_row["value_cpuuser"] if pd.notna(hour_row["value_cpuuser"]) else -np.inf)
    entry["mem_sum"] += hour_row["value_memused"] if pd.notna(hour_row["value_memused"]) else 0.0
    entry["mem_max"] = max(entry["mem_max"], hour_row["value_memused"] if pd.notna(hour_row["value_memused"]) else -np.inf)
    entry["nfs_sum"] += hour_row["value_nfs"] if pd.notna(hour_row["value_nfs"]) else 0.0
    entry["block_sum"] += hour_row["value_block"] if pd.notna(hour_row["value_block"]) else 0.0

    if entry["nhosts"] is None:
        entry["nhosts"] = hour_row["nhosts"]
        entry["ncores"] = hour_row["ncores"]
        entry["timelimit"] = hour_row["timelimit"]
        entry["submit_time"] = to_naive(hour_row["submit_time"])
        entry["start_time"] = to_naive(hour_row["start_time"])


def process_one_file(fpath, cluster, accumulator, capped_jobs):
    df = pd.read_parquet(fpath, columns=NEEDED_COLUMNS)

    hourly = df.groupby("jid").agg(
        nhosts=("nhosts", "max"),
        ncores=("ncores", "max"),
        timelimit=("timelimit", "max"),
        value_cpuuser=("value_cpuuser", "mean"),
        value_memused=("value_memused", "mean"),
        value_nfs=("value_nfs", "mean"),
        value_block=("value_block", "mean"),
        submit_time=("submit_time", "min"),
        start_time=("start_time", "min"),
    ).reset_index()

    for _, row in hourly.iterrows():
        key = (cluster, row["jid"])
        if key in capped_jobs:
            continue

        entry = accumulator.setdefault(key, new_accumulator_entry())
        update_entry_with_hour(entry, row)

        if entry["n"] >= EARLY_WINDOW_SNAPSHOTS:
            capped_jobs.add(key)

    del df, hourly


def process_judge_data():
    accumulator = {}
    capped_jobs = set()

    months = sorted(m for m in os.listdir(JUDGE_ROOT) if os.path.isdir(os.path.join(JUDGE_ROOT, m)))
    print(f"Judge months to process: {months}")

    for month in months:
        month_dir = os.path.join(JUDGE_ROOT, month)
        files = sorted(f for f in os.listdir(month_dir) if f.endswith(".parquet"))
        print(f"  month {month}: processing {len(files)} files...")

        for i, fname in enumerate(files):
            cluster = cluster_code_from_filename(fname)
            fpath = os.path.join(month_dir, fname)
            process_one_file(fpath, cluster, accumulator, capped_jobs)

            if (i + 1) % 200 == 0:
                print(f"    ...{i + 1}/{len(files)} files done "
                      f"(jobs tracked so far: {len(accumulator):,})")

        gc.collect()
        print(f"  month {month}: done. Total unique jobs tracked so far: {len(accumulator):,}")

    return accumulator


def accumulator_to_dataframe(accumulator):
    rows = []
    for (cluster, jid), entry in accumulator.items():
        if entry["n"] == 0:
            continue
        n = entry["n"]
        queue_wait = None
        if entry["submit_time"] is not None and entry["start_time"] is not None:
            try:
                queue_wait = (entry["start_time"] - entry["submit_time"]).total_seconds()
            except Exception:
                queue_wait = None

        rows.append({
            "cluster": cluster,
            "jid": jid,
            "row_id": f"{cluster}_{jid}",
            "nhosts": entry["nhosts"],
            "ncores": entry["ncores"],
            "timelimit": entry["timelimit"],
            "cpu_mean": entry["cpu_sum"] / n,
            "cpu_max": entry["cpu_max"] if entry["cpu_max"] != -np.inf else np.nan,
            "mem_mean": entry["mem_sum"] / n,
            "mem_max": entry["mem_max"] if entry["mem_max"] != -np.inf else np.nan,
            "nfs_mean": entry["nfs_sum"] / n,
            "block_mean": entry["block_sum"] / n,
            "num_early_snapshots": n,
            "queue_wait_seconds": queue_wait,
        })
    return pd.DataFrame(rows)


def main():
    print("=== Step 1: Build early-window features from judge data ===")
    accumulator = process_judge_data()
    job_features = accumulator_to_dataframe(accumulator)
    print(f"\nTotal jobs with features built: {len(job_features):,}")
    print("Cluster breakdown:\n", job_features["cluster"].value_counts())

    print("\n=== Step 2: Load saved per-cluster models ===")
    models = {}
    for cluster in ["S", "C", "Anvil"]:
        path = os.path.join(MODEL_DIR, f"model_{cluster}.joblib")
        if os.path.isfile(path):
            models[cluster] = joblib.load(path)
            print(f"  Loaded model for cluster {cluster}")
        else:
            print(f"  WARNING: no saved model found for cluster {cluster} at {path}")

    print("\n=== Step 3: Predict failure probability per job ===")
    job_features["failure_probability"] = np.nan
    for cluster, model in models.items():
        mask = job_features["cluster"] == cluster
        if mask.sum() == 0:
            continue
        X = job_features.loc[mask, FEATURE_COLUMNS].fillna(0)
        probs = model.predict_proba(X)[:, 1]
        job_features.loc[mask, "failure_probability"] = probs
        print(f"  Cluster {cluster}: predicted {mask.sum():,} jobs, "
              f"mean probability {probs.mean():.3f}")

    print("\n=== Step 4: Merge onto official sample_submission.csv ===")
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print(f"Official required rows: {len(sample_sub):,}")

    merged = sample_sub[["row_id"]].merge(
        job_features[["row_id", "failure_probability"]],
        on="row_id", how="left",
    )

    missing_count = merged["failure_probability"].isna().sum()
    if missing_count > 0:
        # Fallback for any row we couldn't build early-window features for
        # (e.g. a job with zero telemetry rows in the released judge files).
        # Use each row's cluster-specific mean prediction as a reasonable
        # default rather than an arbitrary constant.
        print(f"  {missing_count:,} rows had no buildable features -- "
              f"filling with per-cluster mean probability as a fallback.")
        merged["cluster_from_id"] = merged["row_id"].str.split("_").str[0]
        cluster_means = job_features.groupby("cluster")["failure_probability"].mean()
        for cluster in merged.loc[merged["failure_probability"].isna(), "cluster_from_id"].unique():
            fallback_val = cluster_means.get(cluster, 0.15)  # 0.15 as a last-resort default
            fill_mask = (merged["failure_probability"].isna()) & (merged["cluster_from_id"] == cluster)
            merged.loc[fill_mask, "failure_probability"] = fallback_val
        merged = merged.drop(columns=["cluster_from_id"])

    merged["failure_probability"] = merged["failure_probability"].clip(0, 1)

    print(f"\nFinal submission rows: {len(merged):,}")
    print(merged.head())

    print("\n=== Step 5: Write submission.csv and zip it ===")
    merged.to_csv("submission.csv", index=False)

    with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("submission.csv", arcname="submission.csv")

    print("Wrote submission.csv and submission.zip -- ready to upload to Kaggle.")


if __name__ == "__main__":
    main()