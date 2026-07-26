# Methodology — FRESCO Early Job Failure Prediction

**Team 004** — Prajakta Jadhav, Aariya Vora
**IEEE Computer Society 2026 Global Student Challenge**

## Problem restated

Predict, for each held-out job, the probability that it will FAIL, TIME OUT,
or end in NODE_FAIL (vs. complete successfully), using only information
realistically available early in the job's lifetime. CANCELLED jobs are
excluded from the positive class per competition rules.

Scoring is **macro-averaged AUPRC across three clusters** (Stampede/S,
Conte/C, Anvil) — not accuracy. This requires calibrated probability
outputs and cluster-specific modeling; pooled models are explicitly
insufficient per the competition rules.

## Data leakage: our key correction

Our first draft baseline aggregated **all** hourly snapshots of a job's
lifetime (mean/max CPU, memory, etc. across the job's entire run) to build
features. This is a leakage bug: a job's final hours often show telling
signs of its eventual failure (memory spikes, CPU dropping to zero),
which would only be visible in hindsight — not realistically available
"early" in the job's life as the competition requires.

**Fix:** for every job, we only use its **first `EARLY_WINDOW_SNAPSHOTS`
hourly readings** (currently 2) to build features. This simulates what a
real early-warning system would actually know while the job is still
running.

## Cluster-specific modeling

The raw data encodes cluster identity in filenames:
- `..._S.parquet` → Stampede
- `..._C.parquet` → Conte
- no suffix → Anvil

We train **three completely separate models**, one per cluster, since the
competition explicitly requires cluster-aware routing rather than one
pooled model. Each model only ever trains and predicts on jobs from its
own cluster.

## Feature set (per job, early-window only)

| Feature | Description |
|---|---|
| `nhosts`, `ncores` | Requested resource scale |
| `timelimit` | Requested job time limit |
| `cpu_mean`, `cpu_max` | CPU utilization over the early window |
| `mem_mean`, `mem_max` | Memory usage over the early window |
| `nfs_mean` | NFS (network file system) traffic, early window |
| `block_mean` | Block-device I/O, early window |
| `num_early_snapshots` | How many early hourly readings were actually available (some jobs are shorter than the window) |
| `queue_wait_seconds` | Time between job submission and actual start — known before the job even begins running |

All features come from information available at or before the early
window ends. `exitcode` is used **only** to construct the training label,
never as a model input.

## Memory-safe processing approach

The raw dataset (46,795 hourly Parquet files, ~7.7 GB, tens of millions of
rows) is too large to load into memory in bulk on a standard laptop. We
process **one file at a time**, immediately collapsing it into a running
per-job summary (a small dictionary keyed by `(cluster, jid)`), and discard
the raw rows before moving to the next file. Once a job has accumulated
`EARLY_WINDOW_SNAPSHOTS` readings, we stop updating it, since only the
early window is needed.

## Temporal train/validation split

We train on earlier months and validate on strictly later months, per
cluster era (matching the organizers' own leakage-reduction approach of
separating participant/validation/judge data temporally rather than
randomly):

- Stampede: train Feb–Jun 2013, validate Jul 2013
- Conte: train May/Jun/Aug/Sep/Oct 2015, validate Nov 2016
- Anvil: train Jul–Nov 2022, validate Feb 2023

## Model

`RandomForestClassifier` (scikit-learn), `class_weight="balanced"` to
account for failures being the minority class, trained separately per
cluster. Evaluated using `average_precision_score` (AUPRC) per cluster,
then macro-averaged — replicating the official scoring formula locally.

## Current local results

| Cluster | Validation jobs | Positive rate | AUPRC |
|---|---|---|---|
| Anvil | 32,139 | 0.258 | **0.828** |
| Conte (C) | 104,637 | 0.091 | **0.274** |
| Stampede (S) | 159,433 | 0.148 | **0.348** |

**Macro AUPRC (local estimate): 0.483**

All three clusters score meaningfully above their random-guess baseline
(which equals the positive rate for AUPRC), confirming a genuine, if
early-stage, predictive signal.

## Planned next steps

- Scale up training data (currently a partial month sample per era, not
  the full participant set).
- Try gradient boosting (XGBoost/LightGBM), often stronger than Random
  Forest on tabular data with class imbalance.
- Add richer features: early usage *trend* (rising/falling across the
  early window, not just mean/max), account/queue historical failure
  rates.
- Build and validate the full inference pipeline against the judge set,
  producing `submission.csv` in the required `row_id,failure_probability`
  format.
