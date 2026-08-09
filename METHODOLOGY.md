# Methodology — FRESCO Early Job Failure Prediction

**Team 004** — Prajakta Jadhav, Aariya Vora  
**IEEE Computer Society 2026 Global Student Challenge**

## Problem restated

Predict, for each held-out job, the probability that it will FAIL, TIME OUT, or end in NODE_FAIL (vs. complete successfully), using only information realistically available early in the job's lifetime. CANCELLED jobs are excluded from the positive class per competition rules.

Scoring is macro-averaged AUPRC across three clusters (Stampede/S, Conte/C, Anvil) — not accuracy. This requires calibrated probability outputs and cluster-specific modeling; pooled models are explicitly insufficient per the competition rules.

## Data leakage: our key correction

Our first draft baseline aggregated all hourly snapshots of a job's lifetime to build features — a leakage bug, since a job's final hours often show telling signs of its eventual failure. Fix: for every job, we only use its first 2 hourly readings (`EARLY_WINDOW_SNAPSHOTS = 2`) to build features, simulating what a real early-warning system would know while the job is still running. Per official clarification (Piazza @24), teams choose their own wall-clock horizon **T** and must document it — ours is effectively T ≈ 2 hours, applied consistently across all three clusters.

## Cluster-specific modeling

Cluster identity is encoded in filenames (`_S` = Stampede, `_C` = Conte, no suffix = Anvil). We train three completely separate models, per the competition's explicit cluster-aware routing requirement.

## Feature set (per job, early-window only)

| Feature | Description |
|---|---|
| `nhosts`, `ncores` | Requested resource scale |
| `timelimit` | Requested job time limit |
| `cpu_mean`, `cpu_max` | CPU utilization, early window |
| `mem_mean`, `mem_max` | Memory usage, early window |
| `nfs_mean` | NFS traffic, early window |
| `block_mean` | Block-device I/O, early window |
| `num_early_snapshots` | How many early readings were available |
| `queue_wait_seconds` | Submit-to-start time — known before the job runs |

`exitcode` is used only to construct the training label, never as a model input. Per official clarification (Piazza, scheduler-metadata ruling), fields like `queue`/`account`/`jobname` are also confirmed compliant as early-available features — not yet implemented, listed under future work.

## Memory-safe processing approach

The raw dataset is too large to load into memory in bulk. We process one file at a time, immediately collapsing it into a running per-job summary dictionary, discarding raw rows before the next file. A job stops being updated once it reaches its early-window cap.

## Training data: participant chunk + real validation chunk

We train on the participant chunk (temporal per-cluster-era sampling: Stampede Feb–Jun 2013, Conte May/Jun/Aug/Sep/Oct 2015, Anvil Jul–Nov 2022), plus the officially released validation chunk, which we fold directly into training — explicitly confirmed permitted by organizers (Piazza @46: *"you may include the validation chunk in your training data for your final Kaggle submission"*).

The validation chunk contains real 2017 Stampede and 2023 Anvil labeled data — genuinely closer in time to the actual judge set (2017-04 through 2023-05) than our original 2013/2015/2022 participant-chunk samples. Conte is excluded from this fold-in, since organizers confirmed it was not included in this release (Piazza @51: *"there was an issue producing the Conte data for the validation set"*).

We hold out separate, earlier months (2013-07, 2016-11, 2023-02) as our own local validation set, purely for our own AUPRC estimate — distinct from any data folded into training.

## Model

`HistGradientBoostingClassifier` (scikit-learn), `class_weight="balanced"`, trained separately per cluster. We initially used `RandomForestClassifier` as a baseline; switching to gradient boosting produced a measurable local improvement, most notably on the Conte cluster. Evaluated using `average_precision_score` (AUPRC) per cluster, then macro-averaged.

## Results and honest iteration history

Local validation (held-out 2013/2016/2023 months, not used in training):

| Cluster | Validation jobs | Positive rate | AUPRC |
|---|---|---|---|
| Anvil | 32,139 | 0.258 | 0.830 |
| Conte (C) | 104,637 | 0.091 | 0.319 |
| Stampede (S) | 159,433 | 0.148 | 0.366 |

**Macro AUPRC (local estimate): 0.505**

Real Kaggle public leaderboard progression (official baseline model scores ~0.230 on this same metric):

1. **Early-window fix** (Random Forest, 15-month participant sample): **0.29891**
2. **Model swap to HistGradientBoostingClassifier alone** (no other change): **0.28140** — *a real regression*. Locally this same swap improved macro AUPRC (0.483 → 0.498, driven mostly by Conte), but it did not transfer to the real judge set — a useful, honest finding, most likely because our local validation months (2013/2016) are temporally distant from the 2017/2023 judge period, and job behavior on a real supercomputer plausibly shifts over years.
3. **Folding in the real 2017/2023 validation chunk**: **0.30739** (current best) — the temporally closer real data produced a genuine, reproducible improvement on the actual judge set, not just locally.

This progression is intentionally reported in full, including the step that did not help as expected, because it demonstrates a real and important lesson: local validation performance is only a reliable guide to real performance when the validation data is temporally representative of the actual test distribution. Our 2013/2016 local validation set was not; the 2017/2023 data was, and that is reflected in the results.

## Known open limitation

We do not have a properly temporally-matched local validation set for Stampede/Anvil that mirrors the judge period as closely as the training fold-in does (since folding the validation chunk into training removes it as a held-out local check). Our local AUPRC numbers should therefore be read as a directional signal, not a precise predictor of real leaderboard score — this gap is real and openly acknowledged rather than papered over.

## Planned next steps

- Scale up training data further (still a partial sample per cluster era).
- Add richer features: early usage trend within the window, scheduler metadata (`queue`, `account`, `jobname`) confirmed compliant but not yet implemented.
- Build a properly held-out, temporally-representative local validation set (e.g., a portion of 2017 data reserved rather than fully folded into training) to get a more trustworthy local AUPRC estimate going forward.
