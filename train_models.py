# -*- coding: utf-8 -*-
"""
train_models.py  (rebuilt for low memory use)

This version NEVER loads a full month into memory. Instead, it streams
through files one at a time (each file is only a few thousand rows) and
keeps a small running "accumulator" dictionary -- one tiny entry per job
-- updating it as new files are processed. Raw file data is discarded
immediately after each file is read.

This should use dramatically less RAM than the previous month-batch
version, at the cost of being a bit slower per file (some overhead from
opening many small files individually).

Key correctness properties preserved from before:
- EARLY-WINDOW ONLY: each job's features are built from just its first
  EARLY_WINDOW_SNAPSHOTS hourly readings, never later/final-outcome data.
- THREE SEPARATE CLUSTER MODELS: S, C, Anvil trained independently.
- PROBABILITY OUTPUTS + AUPRC evaluation, matching the real leaderboard.
- TEMPORAL SPLIT: validation months are later in time than training months.
"""

import gc
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_PATH = r"F:\IEEE\kaggle_cache\competitions\fresco-early-job-failure-prediction-challenge"
TRAIN_ROOT = os.path.join(BASE_PATH, "training_data_bundle", "training_data")

MONTHS_FOR_TRAIN = [
    "2013-02", "2013-03", "2013-04", "2013-05", "2013-06",  # S era
    "2015-05", "2015-06", "2015-08", "2015-09", "2015-10",  # C era (2015-07 excluded)
    "2022-07", "2022-08", "2022-09", "2022-10", "2022-11",  # Anvil era
]
MONTHS_FOR_LOCAL_VAL = [
    "2013-07",  # later S month
    "2016-11",  # later C month
    "2023-02",  # later Anvil month
]

EARLY_WINDOW_SNAPSHOTS = 2  # only use each job's first N hourly files

# Only load the columns we actually need -- cuts memory per file substantially
NEEDED_COLUMNS = [
    "time", "submit_time", "start_time", "jid",
    "nhosts", "ncores", "timelimit",
    "value_cpuuser", "value_memused", "value_nfs", "value_block",
    "exitcode",
]

MODEL_OUTPUT_DIR = "models"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cluster_code_from_filename(filename):
    if filename.endswith("_S.parquet"):
        return "S"
    elif filename.endswith("_C.parquet"):
        return "C"
    else:
        return "Anvil"


def to_naive(ts):
    """Strip timezone info if present, so we can safely compare timestamps
    across files that may or may not carry timezone data."""
    if ts is None or pd.isna(ts):
        return ts
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None) if hasattr(ts, "tz_localize") else ts.replace(tzinfo=None)
    return ts


# ---------------------------------------------------------------------------
# Streaming accumulator: one tiny dict entry per (cluster, jid)
# ---------------------------------------------------------------------------
def new_accumulator_entry():
    return {
        "n": 0,
        "cpu_sum": 0.0, "cpu_max": -np.inf,
        "mem_sum": 0.0, "mem_max": -np.inf,
        "nfs_sum": 0.0, "block_sum": 0.0,
        "nhosts": None, "ncores": None, "timelimit": None,
        "submit_time": None, "start_time": None,
        "exitcode": None,
    }


def update_entry_with_hour(entry, hour_row):
    """hour_row is a single row summarizing one job's activity in ONE file
    (one hour), already collapsed across any multiple hosts."""
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
    if entry["exitcode"] is None and pd.notna(hour_row["exitcode"]):
        entry["exitcode"] = hour_row["exitcode"]


def process_one_file(fpath, cluster, accumulator, capped_jobs):
    df = pd.read_parquet(fpath, columns=NEEDED_COLUMNS)

    # Collapse multiple host-rows for the same job within this one hour
    # file into a single summary row per job (mean for usage metrics).
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
        exitcode=("exitcode", "first"),
    ).reset_index()

    for _, row in hourly.iterrows():
        key = (cluster, row["jid"])
        if key in capped_jobs:
            continue  # already have enough early snapshots for this job

        entry = accumulator.setdefault(key, new_accumulator_entry())
        update_entry_with_hour(entry, row)

        if entry["n"] >= EARLY_WINDOW_SNAPSHOTS:
            capped_jobs.add(key)

    del df, hourly
    return


def process_months_streaming(month_list):
    accumulator = {}
    capped_jobs = set()

    for month in month_list:
        month_dir = os.path.join(TRAIN_ROOT, month)
        if not os.path.isdir(month_dir):
            print(f"  month {month}: not found, skipping.")
            continue

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
            "exitcode": entry["exitcode"],
        })
    return pd.DataFrame(rows)


def add_label(features):
    features = features[features["exitcode"] != "CANCELLED"].copy()
    features["target"] = features["exitcode"].apply(
        lambda x: 1 if x in ("FAILED", "TIMEOUT", "NODE_FAIL") else 0
    )
    return features


FEATURE_COLUMNS = [
    "nhosts", "ncores", "timelimit", "cpu_mean", "cpu_max",
    "mem_mean", "mem_max", "nfs_mean", "block_mean",
    "num_early_snapshots", "queue_wait_seconds",
]


def train_and_evaluate(train_features, val_features):
    clusters = sorted(train_features["cluster"].unique())
    per_cluster_auprc = {}

    for cluster in clusters:
        train_c = train_features[train_features["cluster"] == cluster]
        val_c = val_features[val_features["cluster"] == cluster]

        if val_c.empty or train_c.empty:
            print(f"  [{cluster}] skipped -- no train or val data in this sample.")
            continue

        X_train = train_c[FEATURE_COLUMNS].fillna(0)
        y_train = train_c["target"]
        X_val = val_c[FEATURE_COLUMNS].fillna(0)
        y_val = val_c["target"]

        if y_val.nunique() < 2:
            print(f"  [{cluster}] skipped AUPRC -- validation set has only one class present.")
            continue

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, probs)
        per_cluster_auprc[cluster] = auprc

        print(f"  [{cluster}] train_jobs={len(train_c):,} val_jobs={len(val_c):,} "
              f"positive_rate_val={y_val.mean():.3f} AUPRC={auprc:.4f}")

        joblib.dump(clf, os.path.join(MODEL_OUTPUT_DIR, f"model_{cluster}.joblib"))

    if per_cluster_auprc:
        macro_auprc = float(np.mean(list(per_cluster_auprc.values())))
        print(f"\nmacro_cluster_auprc (local estimate) = {macro_auprc:.4f}")
    else:
        print("\nNo clusters had valid AUPRC.")

    return per_cluster_auprc


def main():
    print("Streaming TRAIN months (one file at a time)...")
    train_acc = process_months_streaming(MONTHS_FOR_TRAIN)
    train_features = add_label(accumulator_to_dataframe(train_acc))
    del train_acc
    gc.collect()

    print("\nStreaming LOCAL VALIDATION months...")
    val_acc = process_months_streaming(MONTHS_FOR_LOCAL_VAL)
    val_features = add_label(accumulator_to_dataframe(val_acc))
    del val_acc
    gc.collect()

    print(f"\nTrain jobs: {len(train_features):,} | Validation jobs: {len(val_features):,}")
    print("\nCluster breakdown (train):\n", train_features["cluster"].value_counts())
    print("\nCluster breakdown (validation):\n", val_features["cluster"].value_counts())

    print("\nTraining per-cluster models...")
    train_and_evaluate(train_features, val_features)


if __name__ == "__main__":
    main()