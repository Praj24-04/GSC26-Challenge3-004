# GSC26 Challenge 3 — Data Center Job Failure Prediction

**Team 004** — Prajakta Jadhav, Aariya Vora  
**IEEE Computer Society 2026 Global Student Challenge**  
**Competition:** FRESCO Early Job Failure Prediction Challenge (Kaggle)

## Problem

Predict, for each HPC (high-performance computing) job, the probability that it will FAIL, TIME OUT, or end in NODE_FAIL — using only information realistically available **early** in the job's lifetime, not its full run history or final outcome.

Full technical details, including our data-leakage correction, are in [`METHODOLOGY.md`](./METHODOLOGY.md).

## Dataset — FRESCO

Real job telemetry from three academic HPC clusters (Purdue's Conte and Anvil, and UT Austin's Stampede), released as raw **hourly Parquet files**. Cluster identity is encoded in the filename:
- `..._S.parquet` → Stampede (S)
- `..._C.parquet` → Conte (C)
- no suffix → Anvil

### Target definition
- `0` — job COMPLETED successfully
- `1` — job FAILED, TIMED OUT, or ended in NODE_FAIL
- `CANCELLED` jobs are excluded from training (per competition rules)

### Scoring metric
Leaderboard ranking uses **macro-averaged AUPRC across the three scored clusters** (`macro_cluster_auprc`), not accuracy. We evaluate locally using the same formula.

### Leakage rule followed
`exitcode` is used **only** to construct the label — never as a model input feature. Our first draft had a subtler leakage bug (using a job's entire lifetime instead of just its early portion) — see `METHODOLOGY.md` for how we caught and fixed this.

### Early-window definition
Per official clarification (Piazza @24), teams choose their own wall-clock horizon **T**. We use **T = first 2 hourly telemetry readings per job**, applied consistently across all clusters.

## Cluster-specific modeling requirement

Three separate cluster-specific models (S, C, Anvil), not one pooled model — required per competition rules. `train_models.py` trains and saves one model per cluster independently.

## Pipeline

1. **Stream** raw hourly Parquet files one file at a time (memory-safe).
2. **Accumulate early-window features** per job (first 2 hourly readings).
3. **Label** each job using the exitcode-based target above.
4. **Fold in the official validation chunk** (real 2017 Stampede + 2023 Anvil labeled data, released ~1 week before deadline) into training — explicitly confirmed permitted by organizers (Piazza @46). Conte is excluded from this chunk (confirmed missing by organizers, Piazza @51).
5. **Train** one `HistGradientBoostingClassifier` per cluster (`class_weight="balanced"`), outputting failure **probabilities**.
6. **Evaluate** using AUPRC per cluster, then macro-average.
7. **Generate inference** (`generate_submission.py`): streams the judge set the same memory-safe way, routes each job to its cluster's model, and produces `submission.csv` in the exact `row_id,failure_probability` format required by `sample_submission.csv`.

## Current results

**Local validation (held-out 2013/2016/2023 months):**

| Cluster | Validation jobs | Positive rate | AUPRC |
|---|---|---|---|
| Anvil | 32,139 | 0.258 | 0.830 |
| Conte (C) | 104,637 | 0.091 | 0.319 |
| Stampede (S) | 159,433 | 0.148 | 0.366 |

**Macro AUPRC (local estimate): 0.505**

**Real Kaggle public leaderboard score: 0.30739** (public leaderboard uses 5% of judge data; official baseline model scores ~0.230, so this represents a genuine improvement over baseline).

Progression across iterations, each a real, verified improvement:
1. Initial submission (whole-lifetime features, leakage bug): not submitted after correction
2. Early-window fix + Random Forest, 15-month sample: **0.28140**
3. Model swap to HistGradientBoostingClassifier: (regression noted, see Methodology)
4. + Real 2017/2023 validation chunk folded into training: **0.30739** (current best)

## What's next (honest, real remaining gaps)

- Scale up training data further (currently a partial month sample per cluster era, not the full 46,795-file participant set).
- Add richer features: early usage *trend* within the window, historical account/queue failure rates, scheduler metadata (`queue`, `account`, `jobname` — confirmed compliant to use per Piazza clarification, not yet implemented).
- Investigate temporal distribution shift more rigorously — our local validation (2013/2016) does not perfectly predict real judge-set (2017) performance, a known limitation documented transparently here.

## Files

- `train_models.py` — full training pipeline (streaming, early-window features, validation-chunk folding, per-cluster model training, AUPRC evaluation).
- `generate_submission.py` — inference pipeline: builds judge-set features, applies saved models, produces `submission.csv`.
- `METHODOLOGY.md` — detailed write-up of our approach and the leakage bug we found and fixed.
- `requirements.txt` — dependencies.
- `models/` — saved per-cluster model files.

## Note on data

Per competition rules, this repository does **not** include the FRESCO competition data itself. Scripts expect the raw Parquet files downloaded separately (via `kagglehub`) and placed at the path configured at the top of each script.
