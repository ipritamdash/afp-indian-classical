# Pre-Registration: Out-of-Library Threshold Calibration for Recipe v3 Demo

**Locked:** 2026-05-15, BEFORE any out-of-library scores are measured.

## 1. Hypothesis

There exists a confidence threshold T on the top-1 sequence-similarity score
(mean cosine over query segments) such that:

- **In-library queries** (truth is among the 357 Saraga refs) score above T with
  high probability — i.e., are NOT falsely rejected
- **Out-of-library queries** (truth is NOT in the Saraga library) score below T
  with high probability — i.e., are correctly rejected as "no match"

The two distributions are separable enough that a single scalar threshold yields
acceptable error rates for an interactive demo.

## 2. Model under calibration

- **Recipe v3 seed 42** ckpt-30 (the primary seed per `recipe_v3_30ep/PROTOCOL.md`)
- Encoder + FAISS pipeline identical to `scripts/nafp/recipe_v2_eval/eval.py`
- This is the model that will be deployed in the public HF Spaces demo
- Baseline NAFP-ckpt-10 is NOT calibrated in this run (different deployment scope)

## 3. In-library score distribution (already measured)

Source: `data/results/nafp/recipe_v3_30ep/seed42_eval/main_1s/query_results.parquet`

For each of 1,000 main_1s queries, we read the rank-1 `nafp_score` (the mean
cosine over the query's sequence segments matched against the candidate ref).
**Restrict to the 995 queries where rank-1 is a CORRECT match** (HR@1 = 0.995).
This is our "true positive" score distribution.

The 5 wrong-rank-1 misses are out-of-scope here (they're "in-library queries the
model fails on" — a separate failure mode addressed by Recipe v3 already).

## 4. Out-of-library score distribution (to be measured)

**Probe set: 200 random clips drawn from FMA-medium.**

- Source: `data/fma/fma_medium.zip` (Free Music Archive medium, ~25k Western tracks)
- Selection: sample 200 unique MP3 paths from the zip's index with a
  reproducible seed (`np.random.default_rng(20260515)`)
- Per-track: load full audio at 8 kHz mono, pick a random 1-second offset
  uniformly from [0, duration − 1] with the same seed
- Encode through recipe v3 seed 42, search the existing 690,414-segment FAISS
  index over Saraga refs, record rank-1 `nafp_score`

### Why FMA is a valid OOL probe (even though the encoder was trained on FMA)

The encoder was trained on a 10k-track FMA subset (`train-10k-30s/`). Some of
the 200 probe tracks may or may not have been in that training subset.

**This is intentionally not controlled for**, because:
- The retrieval library is **Saraga (357 Indian classical tracks)**, NOT FMA.
- An "out-of-library" query is one whose truth is not in the Saraga library.
- An FMA track has no truth in the Saraga library by construction.
- Whether the encoder saw the track during training is irrelevant to the
  question "is the truth in the library."

The calibration measures **whether the model's top-1 score reliably distinguishes
'truth exists in Saraga library' from 'truth does not exist in Saraga library'**,
which is the deployment question.

### What the OOL probe set does NOT cover

- Indian classical music **not** in our 357-track Saraga subset (e.g., Bollywood,
  CompMusic CMD/HMD, Sanidha) — harder OOL case; we do not have legal-grade
  copies of these in bulk
- Heavily noisy / phone-mic recordings — separate failure mode
- Same-artist different-work confusions where the work is NOT in the library —
  may produce false-positive matches because of the same-artist hub signal

These limitations will be disclosed in the demo's "About" tab.

## 5. Threshold-selection rule (pre-registered)

Locked criterion: **pick the smallest T such that FPR ≤ 0.05 on the OOL probe set.**

- FPR (false-positive rate) = fraction of OOL queries with top-1 score ≥ T
- After picking T, report TPR (true-positive rate) on the in-library set =
  fraction of in-library correct matches with top-1 score ≥ T
- Also report TPR at a tighter operating point (FPR ≤ 0.01) for a "high
  confidence" badge

## 6. Falsification rule

The calibration is declared **infeasible** if no T achieves both:
- FPR (OOL falsely accepted) ≤ 0.05 on the 200 probes
- TPR (in-library correctly accepted) ≥ 0.95 on the in-library set

If infeasible: the demo will not surface a "no match" button; it will display
top-5 candidates only, with a generic disclaimer that out-of-library inputs
will produce false matches.

## 7. Statistical method

- Score histograms for both distributions (60 bins, [0, 1])
- ROC curve (TPR vs FPR sweep over T ∈ [0, 1])
- Wilson 95% CI on TPR / FPR at the chosen operating point

## 8. Reporting

Output files in `data/results/threshold_calibration/`:
- `PROTOCOL.md` (this file) — locked before measurement
- `ool_scores.csv` — per-probe (track_id, offset_sec, top1_score, top1_ref_id)
- `histograms.png` + `roc.png` — score distributions + ROC
- `RESULTS.md` — chosen T, TPR/FPR at chosen + tight operating points, with CI
- `threshold.json` — machine-readable `{T_default, T_high_conf}` for demo code

## 9. Computation

- Sampling seed: `20260515`
- FMA extraction: stream from zip (no need to disk-extract all 22 GB)
- Encoder env: `TF_USE_LEGACY_KERAS=1`, recipe_v3 seed 42 ckpt-30
- Hardware: Mac M5 Metal (verified faster than CPU for this workload)

Locked. No retroactive changes to threshold rule, probe-set size, or seed.
