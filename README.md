# GSC26 Challenge 3 — Data Center Job Failure Prediction

**Team 004** — Prajakta Jadhav, Aariya Vora
**IEEE Computer Society 2026 Global Student Challenge**

## Problem

Predict whether an HPC (high-performance computing) job will fail, time out, or
end in a node failure, using only early job metadata and resource-usage signals
— not the job's final outcome. Early, reliable prediction lets a scheduler
intervene before wasting hours of expensive compute time on a job that's
already doomed.

## Dataset — FRESCO

We use the FRESCO dataset: real job telemetry collected from three academic
HPC clusters (Purdue's Conte and Anvil, and UT Austin's Stampede). Data is
released as raw **hourly Parquet files** — each row is one hourly snapshot of
one job on one host, not one row per job. Cluster identity is encoded in the
filename:
- `..._S.parquet` → Stampede
- `..._C.parquet` → Conte
- no suffix → Anvil

### Target definition
- `0` — job COMPLETED successfully
- `1` — job FAILED, TIMED OUT, or ended in NODE_FAIL
- `CANCELLED` jobs are excluded from training entirely (per competition rules)

### Leakage rule followed
`exitcode` and any other field revealing the final outcome are used **only**
to construct the label — never as a model input feature.

## Pipeline (current baseline)

1. **Load** a sample of raw hourly Parquet files across several months.
2. **Aggregate** all hourly rows belonging to the same job (`jid`) into a
   single row per job, summarizing resource usage (mean/max CPU, memory,
   NFS, block I/O) and static job attributes (requested hosts, cores, time
   limit).
3. **Label** each job using the exitcode-based target defined above.
4. **Train** a Random Forest baseline classifier on the aggregated
   per-job features.
5. **Evaluate** using precision, recall, and F1 — plain accuracy is
   misleading here since failures are the minority class.

## Current baseline results

On a sample of ~74k jobs (3 months, partial file sample per month):

| Metric | Completed | Failed / Timeout / Node-Fail |
|---|---|---|
| Precision | 0.86 | 0.79 |
| Recall | 0.94 | 0.58 |
| F1-score | 0.90 | 0.67 |

Overall accuracy: 0.85

**Top predictive features:** mean/max memory usage, requested time limit,
CPU usage — matches real-world intuition (jobs failing from memory
exhaustion or hitting their time limit).

## What's next

- Scale up training data (currently using a small file sample per month
  for fast iteration — full dataset has 46,795 participant files).
- Improve recall on the failure class (currently missing ~42% of real
  failures) — likely needs better features (e.g., trends over time within
  a job, not just mean/max) or class-imbalance handling beyond
  `class_weight="balanced"`.
- Build proper temporal train/validation split (per competition rules,
  avoid random splits that leak future behavior into training).
- Generate predictions for the judge set in the required
  `row_id = <cluster>_<jid>` submission format.

## Files

- `train_baseline.py` — data loading, aggregation, and baseline model
  training pipeline described above.
