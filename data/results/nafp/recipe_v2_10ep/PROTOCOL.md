# Pre-Registration: NMFP-Recipe Retrain @ 10 Epochs

**Locked: 2026-05-14, BEFORE any training run.**

## 1. Hypothesis

Applying two of NMFP's (Araz et al., ISMIR 2025, arxiv 2506.22661) published
recipe fixes to NAFP training at the same 10-epoch budget as our existing
baseline will reduce the same-artist intra-class confusion failure mode on
Saraga 1.5 main_1s queries.

Specifically:
1. **F_MIN: 300 → 160 Hz** — captures the 160-300 Hz band where Indian classical
   music drone (tanpura), vocal fundamentals, and tabla low harmonics live.
2. **One-anchor-per-track sampler** (`seg_mode='random_oneshot'` re-sampled per
   epoch) — removes in-batch false negatives that arise when multiple segments
   from the same source track are present in one mini-batch and pushed apart by
   NT-Xent's denominator.

NT-Xent loss preserved (τ=0.05). BSZ=120 preserved. Architecture (NAFP nnfp.py)
preserved. Augmentation pool preserved. Training data (FMA-medium 10k_icassp)
preserved. Optimizer (Adam, LR=1e-4, cos schedule) preserved.

The only changes are the two NMFP fixes above. Cite Araz et al. for both.

## 2. Baseline

- Model: NAFP-ckpt-10 (trained by us, 10 epochs, FMA-medium 10k_icassp)
- Evaluation: Saraga 1.5, 8 cells (main + ablation × {1s, 3s, 5s, 10s})
- Headline: HR@1 main_1s = 0.983 (983/1000), 17 misses, 14/17 same-artist
  top-1 confusions

## 3. New model

- Name: `NAFP-NMFPrecipe-ckpt-10` (or `recipe_v2_10ep` in the directory tree)
- Training: 10 epochs, FMA-medium 10k_icassp, Adam LR=1e-4 cosine,
  BSZ=120 with N_ANCHORS=60 N_POSITIVES_PER_ANCHOR=2, NT-Xent τ=0.05
- Recipe deltas vs baseline:
  - `MODEL.F_MIN: 300 → 160`
  - `seg_mode: 'all' → 'random_oneshot'` with epoch-level re-sampling

## 4. Primary endpoint

**HR@1 on Saraga main_1s.**

Decision rule: report as success if BOTH of the following hold:
- HR@1 new_model > 0.983 (any positive Δ)
- McNemar exact two-sided binomial test on paired 1000 queries: p < 0.05
  - With b = (new-hit AND baseline-miss) and c = (new-miss AND baseline-hit),
    significance at unadjusted α=0.05 (1-sided) requires b ≥ 6 if c=0;
    b ≥ 7 if c=1; b ≥ 8 if c=2.

Bonferroni correction at α/8 = 0.00625 NOT required for headline (8-cell
correction would force b ≥ 8 with c=0, which is a higher bar; we pre-register
unadjusted on the primary cell only, with secondary cells reported as
descriptive).

## 5. Secondary endpoints

- HR@1 on main_3s, main_5s, main_10s, ablation_{1,3,5,10}s — descriptive only
- Falsification: no cell may regress by > 1 miss (i.e., HR@1 drop > 0.001
  for main cells, > 0.0016 for ablation cells)
- Same-artist confusion among remaining main_1s misses: must drop from
  14/17 baseline ratio by ≥ 30% (i.e., to ≤ 9 same-artist misses, regardless
  of total miss count)

## 6. Statistical method

McNemar's exact test on paired binary outcomes per cell. No multi-seed
factorial (single seed=42, fixed). Discordant pair counts (b, c) reported
per cell. Pre-registered p-value threshold: unadjusted α=0.05 on primary.

## 7. Falsification

If primary endpoint fails (HR@1 ≤ 0.983 OR p ≥ 0.05): report as third
pre-registered negative result, joining Intervention 2 and hubness
post-processing. No post-hoc cell-switching or hypothesis modification.

If primary passes but any cell regresses > 1 miss: report as confounded;
the gain came with a trade-off, not a clean improvement.

## 8. Compute

- Platform: Google Colab GPU (T4 or better)
- Wall time budget: ≤ 90 minutes for full training run
- Single seed (random_seed=42, set in tf.keras.utils.set_random_seed)
- No mid-training intervention; training runs to ckpt-10 unattended

## 9. License + attribution

- F_MIN=160 and one-anchor-per-track sampler are from Araz et al. ISMIR 2025
  (arxiv 2506.22661, github.com/raraz15/neural-music-fp)
- We implement these in our NAFP fork from the paper description (NOT by
  copying NMFP code). Cite Araz et al. in the writeup.
- Our trained ckpt-10 is our weights, under our existing repo license
  (MIT-compatible NAFP-upstream lineage).
- No NMFP weights, code, or training data are used.

## 10. Artifacts

After training + evaluation:
- `scripts/nafp/upstream/config/default_recipe_v2.yaml` — modified config
- `scripts/nafp/upstream/model/utils/dataloader_keras.py` — modified sampler
- `data/results/nafp/recipe_v2_10ep/ckpt-10.{index,data-00000-of-00001}` — trained weights
- `data/results/nafp/recipe_v2_10ep/scores_{cell}.json` — eval results per cell
- `data/results/nafp/recipe_v2_10ep/mcnemar_results.csv` — significance per cell
- `data/results/nafp/recipe_v2_10ep/RESULTS.md` — final writeup

Locked. No retroactive modifications to primary endpoint or threshold.
