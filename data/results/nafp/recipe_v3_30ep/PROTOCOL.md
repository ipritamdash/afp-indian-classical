# Pre-Registration: NAFP+NMFPrecipe v3 — 30 Epochs × 3 Seeds

**Locked: 2026-05-14, BEFORE any v3 training run.**

## 1. Hypothesis

Following recipe_v2's borderline result (HR@1=0.991, McNemar p=0.152, n.s.),
we scale UP two dimensions while preserving the mechanism:

1. **Epochs: 10 → 30.** Allow the recipe more time to converge and reduce the
   "broke 8 previously-correct" noise observed at 10 ep.
2. **Batch size: 120 → 320 (N_ANCHOR 60 → 160).** Provides 2.7× more in-batch
   negatives per NT-Xent gradient step, directly compensating for the
   one-anchor-per-track sampler's reduced positive density per epoch.
3. **3 seeds (42, 137, 2026).** Pooled McNemar across 3000 paired observations
   to distinguish real recipe effect from single-seed stochasticity.

Recipe fixes preserved from recipe_v2:
- `F_MIN: 160` (NMFP fix #6, Araz et al. ISMIR 2025)
- `TR_SEG_MODE: random_oneshot` with per-epoch resampling (NMFP fix #2)

Loss preserved as NT-Xent (τ=0.05). Architecture unchanged. Training data
unchanged (FMA-medium 10k_icassp). Optimizer Adam, LR=1e-4, cos schedule.

Epoch parity with baseline (10 ep) intentionally broken to provide convergence
headroom. Writeup will disclose this explicitly: "we trained 3× the baseline
budget to give the recipe time to express its benefits."

## 2. Baseline

- Model: NAFP-ckpt-10 (FMA-medium 10k_icassp, 10 ep, BSZ=120, NT-Xent τ=0.05)
- HR@1 main_1s = 0.983 (17 misses, 14/17 same-artist confusions)

## 3. New model: NAFP-NMFPrecipe-v3-ckpt-30 (3 seeds)

Seeds: 42 (primary), 137 (replication-1), 2026 (replication-2).

## 4. Primary endpoint

**HR@1 on Saraga main_1s, POOLED across all 3 seeds (3000 paired observations).**

Decision rule — recipe v3 is declared a success iff BOTH:
1. Mean HR@1 across 3 seeds > 0.983
2. Pooled McNemar exact 2-sided binomial p < 0.05 unadjusted
   - Pooled b = sum of per-seed b (baseline-hit, v3-miss)
   - Pooled c = sum of per-seed c (baseline-miss, v3-hit)
   - p_exact = 2 × binom_cdf(min(b,c), b+c, 0.5)

Stronger threshold (Bonferroni at 8 cells): p < 0.00625. Reported if achieved
but not required.

## 5. Per-seed reporting (descriptive)

For each seed, report:
- HR@1 main_1s
- McNemar b, c, p (per-seed, unadjusted)
- Same-artist confusion count among remaining misses

Used to characterize seed-to-seed variance, not as primary endpoint.

## 6. Secondary endpoints (descriptive only)

Across the 3 seeds:
- HR@1 main_3s, main_5s, main_10s, ablation_{1,3,5,10}s
- Falsification: no cell may regress by >1 miss on AVERAGE across 3 seeds

## 7. Falsification

If primary endpoint fails (mean HR@1 ≤ 0.983 OR pooled p ≥ 0.05): report as
fourth pre-registered negative result, joining Intervention 2, hubness postproc,
and recipe v2. No post-hoc cell switching or seed cherry-picking.

If primary passes but any cell regresses >1 miss: report as confounded.

## 8. Compute

- Platform: Google Colab L4 GPU (single ~5h session)
- Per-seed training: ~90 min wall (BSZ=320 × 30 epochs)
- Sequential within one notebook: seed 42 → seed 137 → seed 2026
- Random seed set via `tf.keras.utils.set_random_seed(SEED)` before training
- Single notebook, three separate Run-All triggers (one per seed)

## 9. License + attribution

- F_MIN=160 + one-anchor-per-track sampler from Araz et al. ISMIR 2025
  (arxiv 2506.22661). Implemented from paper description, cite in writeup.
- NT-Xent loss is NAFP's original (Chang et al. ICASSP 2021).
- All 3 trained checkpoints are our weights, under repo's MIT-compatible license.
- No NMFP weights, code, or training data used. Same FMA-medium as baseline.

## 10. Artifacts

After training + evaluation:
- `recipe_v3.yaml` — modified config
- `ckpt-30_seed{42,137,2026}.{index,data-00000-of-00001}` — 3 checkpoints
- `query_results_seed{42,137,2026}.parquet` per cell × 3 seeds
- `pooled_mcnemar.csv` — primary endpoint computation
- `RESULTS.md` — final writeup

Locked. No retroactive modifications.
