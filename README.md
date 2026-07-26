# GSC26 Challenge 3 — Data Center Job Failure Prediction

**Team 004** — Prajakta Jadhav, Aariya Vora
**IEEE Computer Society 2026 Global Student Challenge**
**Competition:** FRESCO Early Job Failure Prediction Challenge (Kaggle)

## Problem

Predict, for each HPC (high-performance computing) job, the probability
that it will FAIL, TIME OUT, or end in NODE_FAIL — using only information
realistically available **early** in the job's lifetime, not its full
run history or final outcome. Early, reliable prediction lets a scheduler
intervene before wasting hours of expensive compute time on a job that's
already doomed.

Full technical details, including our key data-leakage correction, are in
[`METHODOLOGY.md`](./METHODOLOGY.md).

## Dataset — FRESCO

Real job telemetry from three academic HPC clusters (Purdue's Conte and
Anvil, and UT Austin's Stampede), released as raw **hourly Parquet
files** — each row is one hourly snapshot of one job on one host, not one
row per job. Cluster identity is encoded in the filename:
- `..._S.parquet` → Stampede (S)
- `..._C.parquet` → Conte (C)
- no suffix → Anvil

### Target definition
- `0` — job COMPLETED successfully
- `1` — job FAILED, TIMED OUT, or ended in NODE_FAIL
- `CANCELLED` jobs are excluded from training (per competition rules)

### Scoring metric
Leaderboard ranking uses **macro-averaged AUPRC across the three scored
clusters** (`macro_cluster_auprc`), not accuracy. We evaluate locally
using the same formula.

### Leakage rule followed
`exitcode` and any field revealing the final outcome are used **only** to
construct the label — never as a model input feature. Our first draft
had a subtler leakage bug (using a job's *entire* lifetime to build
features, not just its early portion) — see `METHODOLOGY.md` for how we
caught and fixed this.

## Cluster-specific modeling requirement

This challenge requires **three separate cluster-specific models**
(S, C, Anvil), not one pooled model — a pooled model without
cluster-aware routing is explicitly insufficient per competition rules.
`train_models.py` trains and saves one model per cluster independently.

## Pipeline

1. **Stream** raw hourly Parquet files one file at a time (memory-safe —
   never loads a full month into memory at once).
2. **Accumulate early-window features** per job: only the job's first
   `EARLY_WINDOW_SNAPSHOTS` (currently 2) hourly readings are used —
   simulating what would realistically be known while a job is still
   running.
3. **Label** each job using the exitcode-based target above.
4. **Split temporally**: train on earlier months, validate on strictly
   later months, per cluster era (no random shuffling across time).
5. **Train** one `RandomForestClassifier` per cluster
   (`class_weight="balanced"`), outputting failure **probabilities**.
6. **Evaluate** using AUPRC per cluster, then macro-average — matching
   the official leaderboard formula.

## Current results (local validation)

| Cluster | Validation jobs | Positive rate | AUPRC |
|---|---|---|---|
| Anvil | 32,139 | 0.258 | **0.828** |
| Conte (C) | 104,637 | 0.091 | **0.274** |
| Stampede (S) | 159,433 | 0.148 | **0.348** |

**Macro AUPRC (local estimate): 0.483**

All three clusters score meaningfully above their random-guess baseline
(which equals the positive rate, for AUPRC), confirming genuine
predictive signal from early-window features alone.

## What's next

- Scale up training data (currently a partial month sample per cluster
  era, not the full 46,795-file participant set).
- Try gradient boosting (XGBoost/LightGBM) as a stronger alternative to
  Random Forest.
- Add richer features: early usage *trend* within the window (not just
  mean/max), historical account/queue failure rates.
- Build the inference pipeline for the judge set and generate
  `submission.csv` in the required `row_id,failure_probability` format.

## Files

- `train_models.py` — full training pipeline described above (data
  streaming, early-window feature extraction, per-cluster model
  training, AUPRC evaluation).
- `METHODOLOGY.md` — detailed write-up of our approach, including the
  leakage bug we found and fixed.
- `requirements.txt` — dependencies needed to run `train_models.py`.
- `models/` — saved per-cluster model files (if included; otherwise
  regenerate by running `train_models.py`).

## Note on data

Per competition rules, this repository does **not** include the FRESCO
competition data itself. `train_models.py` expects the raw Parquet files
to be downloaded separately (via `kagglehub` or the Kaggle competition
page) and placed at the path configured at the top of the script.
