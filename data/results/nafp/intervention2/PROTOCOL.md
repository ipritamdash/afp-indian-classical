# Intervention 2 — Per-Artist Mean Subtraction at NAFP Inference

**Pre-registered:** 2026-05-12 — *before* any sweep was run.

This file is the locked-in protocol. Any change after this point must be explicit and
dated; comparing only the *pre-registered* numbers protects against p-hacking.

---

## 1. Hypothesis

NAFP main-1s misses are dominated by same-artist top-1 confusions:
empirically **14 / 17** misses (82 %) on Saraga main 1 s have the top-1 ref by the
same artist as the truth (verified 2026-05-12 against
`data/results/nafp/saraga_only_main_1s/query_results.parquet`).

Hypothesis: there is a per-artist "hub direction" in NAFP embedding space — a
component shared by all segments of a single artist regardless of raaga / work.
Subtracting this hub partially from ref embeddings de-confounds artist identity
from segment identity and reduces same-artist top-1 confusions.

## 2. Variant

**V1 — Asymmetric per-artist mean subtraction.** For each ref segment `e_seg`
belonging to ref `r` with artist `a`:

```
e_seg'  = (e_seg − α · μ_a^{LOO(r)}) / ‖e_seg − α · μ_a^{LOO(r)}‖
```

where `μ_a^{LOO(r)}` is the mean of all segments of all refs in artist `a`
**excluding ref r's own segments** (leave-one-track-out — mandatory; standard
AS-Norm protocol, Matejka 2017).

Query embeddings are **not** modified — the test-time query has no artist label.
This makes the score asymmetric, accepted: at test time the only knowable artist
is the candidate's.

Pre-computation: `centroids_per_ref.npy` shape (357, 128) — one LOO centroid per
ref. Built by `scripts/nafp/intervention2/build_centroids.py` 2026-05-12. Sanity
verified.

## 3. Variants rejected and why

- **k-reciprocal re-ranking** — degenerate at L=1 (need a meaningful neighborhood)
- **Cascade with k-reciprocal** — inherits the above
- **Mutual Proximity** — too aggressive for 357-ref library; back-burner robustness check
- **AS-Norm score-level** — fallback variant if V1 fails Bonferroni-corrected significance
- **Per-(artist, raaga)** — would subtract raaga signal, which is legitimate musical content

## 4. Hyperparameter

**Headline α (pre-registered): α = 0.10.**

Rationale: AS-Norm literature reports α in [0.05, 0.30] for similar speaker /
content hub corrections (Kenny 2010; Matejka 2017). 0.10 is the geometric
midpoint that does not pull embeddings substantially off the unit sphere.

**Sweep range:** {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.
Bounded above at 0.30: empirically, α > 0.30 lengths centroid_norm × α >
~0.20 against a unit-norm embedding, which begins to dominate the embedding
geometry (over-subtraction).

α = 0 is a built-in sanity control: the rescore path must reproduce baseline
HR@1 / MRR bit-identically. Failing this assertion aborts the sweep.

## 5. Cells

**Primary (4 cells):** NAFP Saraga main × {1 s, 3 s, 5 s, 10 s}.
**Secondary (4 cells):** NAFP Saraga ablation × {1 s, 3 s, 5 s, 10 s}.

Total: **8 cells**.

## 6. Controls

Per cell, three control conditions in addition to the α-sweep:

1. **α = 0** (identity sanity) — must equal baseline bit-for-bit
2. **Shuffled-centroid** (seed = 20260512) — assign each ref a centroid from a
   randomly-chosen *other* artist. If V1 gain is "real artist hub", shuffled
   should not help (and may hurt). If shuffled helps equally, the mechanism is
   not what we claim.
3. **Isotropic-centroid** — replace each centroid with a unit-Gaussian random
   vector of matching norm. Tests whether *any* centroid subtraction helps via
   anti-hub effect (Mu & Viswanath 2018 all-but-the-top) or whether
   artist-specific structure is essential.

Controls run at α = α* (cross-corpus winner; see §7) only — they exist to
characterise mechanism at the chosen operating point.

## 7. Hyperparameter selection (no peeking)

**Cross-corpus tuning.** Tune α on Hindustani-only, evaluate on Carnatic-only;
report **both** symmetric directions:
- H → C: pick α* = argmax HR@1 on Hindustani; report HR@1 on Carnatic at α*
- C → H: pick α* = argmax HR@1 on Carnatic; report HR@1 on Hindustani at α*
- Pooled: report HR@1 on pooled Saraga at α = 0.10 (the literature-pinned
  headline; not data-derived)

This prevents "the best α on the test set" cherry-picking. Held-out HR@1 is the
defensible number.

## 8. Statistical test

**McNemar's exact test** (Dietterich 1998) on paired binary outcomes (hit / miss)
per query. Tests asymmetry of (baseline-hit, intervention-miss) vs
(baseline-miss, intervention-hit) discordant pairs.

**Multiple-comparison correction:** Bonferroni across 8 cells.
α_family = 0.05 / 8 = **0.00625**.

A cell is declared **significant** if the McNemar exact two-sided p-value at
α = 0.10 (the pre-registered headline) is < 0.00625.

**Asymmetric regression policy** (Agent B convention):
- Gains: must clear Bonferroni-corrected significance to count
- Regressions: must be non-significant to be acceptable
  (i.e. a single significant regression is grounds to ship V1 only at lengths
  where gain is significant AND no regression exists)

## 9. Reporting

For each of 8 cells, report:
- baseline HR@1 + Wilson 95 % CI + n_hits / n
- intervention HR@1 @ α = 0.10 + Wilson 95 % CI
- ΔHR@1 (intervention − baseline) + 95 % bootstrap CI (5000 resamples,
  seed = 20260513)
- McNemar exact p-value
- Bonferroni-significant flag
- Top-1 same-artist confusion rate among miss queries (the targeted metric)

Plus risk-coverage curve over α ∈ sweep for the 4 primary main cells.

Plus the 3 control results at the cross-corpus winner α*.

Plus a per-artist breakdown table (≥ 5-ref artists) showing how the intervention
shifts each artist's HR@1.

## 10. Decision rule

V1 is **adopted** for the paper if AND ONLY IF, at α = 0.10:
1. ≥ 4 of 4 primary main cells show ΔHR@1 ≥ 0
2. ≥ 2 of 4 primary main cells clear McNemar Bonferroni
3. No primary cell shows a Bonferroni-significant regression
4. Shuffled-centroid control shows substantially smaller (or negative) ΔHR@1
   than V1

Failing (1) or (3): V1 is reported as a negative result. We disclose it,
explain the failure mode in the paper Limitations section, and pivot to AS-Norm
(V2, fallback) per the pre-registered fallback path.

Failing (4): mechanism is not the claimed one; report findings honestly, do not
claim "per-artist hub removal" — the gain (if any) is generic centroid effect.

## 11. Artifacts

Each run produces:
- `data/results/nafp/intervention2/<cell>/scores_alpha_<a>.json`
- `data/results/nafp/intervention2/<cell>/query_results_alpha_<a>.parquet`
- `data/results/nafp/intervention2/<cell>/scores_control_<name>.json`
- `data/results/nafp/intervention2/analysis/mcnemar_results.csv`
- `data/results/nafp/intervention2/analysis/SUMMARY.md`

## 12. Out of scope

- The 23 "right-ref wrong-offset" alignment outliers (NAFP `align_err > 0.5 s`):
  separate failure mode (segment selection within correct ref), orthogonal to
  V1. Not addressed here.
- Query-side mean subtraction (would require oracle artist label at test time).
- Re-training. V1 is inference-only.
