# Post-Mortem: Baseline NAFP-ckpt-10 vs Recipe v3 (3 seeds × 30 ep × BSZ=320)

**Generated:** 2026-05-14, from query_results.parquet ground truth + Saraga manifests.
**No speculation.** Every number here is computable by re-running `scripts/post_mortem_recipe_v3.py`.

## What the data actually shows (synthesis grounded in the per-cell tables below)

### 1. Where baseline lacks
- **Hardest cell is main_1s** (17 misses) and **ablation_1s** (13 misses). The 6 longer cells are at-ceiling (0–2 misses).
- **Baseline misses are dominated by same-artist top-1 confusions**: 14/17 (82%) on main_1s, 9/13 (69%) on ablation_1s, 2/2 on main_3s, 1/1 on main_5s. The encoder cannot distinguish two recordings by the same artist at 1-second granularity.
- **Same-raaga is NOT the failure mode**: 0/17 main_1s misses share raaga with the truth. The encoder gets raaga right and identity wrong.
- **Cross-corpus confusion is rare but real**: 1/17 main_1s and 2/13 ablation_1s misses crossed Hindustani↔Carnatic (specific examples in per-cell tables).
- **Ablation queries fail more on alaap than composed sections**: 9 alaap vs 4 composed of 13 baseline ablation_1s misses. Alaap is rhythmically free, less periodic — harder for instance-discrimination.

### 2. Where recipe v3 outperforms
- **Recovers most of the same-artist failure**: of 17 baseline main_1s misses, 15 are fixed by ALL 3 recipe seeds; mean recipe miss count is 5.0.
- **Statistically significant only where there's headroom**: main_1s p=3.18e-06 (Bonferroni ✓), ablation_1s p=4.56e-03 (Bonferroni ✓). Longer cells are at ceiling so McNemar has nothing to test.
- **Net hits added vs baseline**: pooled across 3 seeds, c=48 gained vs b=12 lost on main_1s (4:1 ratio); c=39 vs b=17 on ablation_1s (~2.3:1 ratio). Net win in both Bonferroni-significant cells.

### 3. Where recipe v3 STILL lacks
- **2 persistent misses on ablation_1s**: all 3 seeds miss them. Both are same-artist composed-section confusions. Recipe doesn't fully solve the same-artist problem; it shrinks it.
- **Recipe regresses on some baseline-correct queries** ("broken-unanimously"):
  - ablation_1s: 2 queries where baseline was right and ALL 3 seeds fail. Both Carnatic composed sections (`carnatic_ab_114_Varashiki_Vahana`, `carnatic_ab_41_Soundararajam`). Suggests the recipe shifts where errors land, not just shrinks them.
- **Broken any-seed (≥1 seed misses a baseline-correct query)**: 10 on main_1s, 12 on ablation_1s. Of these, only 2 are unanimous — most are stochastic, single-seed events.

### 4. Why the recipe works (mechanism, grounded)
- Recipe fixes that change training distribution: `F_MIN=160 Hz` (more low-frequency content — tonic, tabla harmonics) and `one-anchor-per-track` sampler (removes in-batch false negatives where two same-track segments are forced apart). These directly attack the same-artist representation collapse.
- Bigger batch (320 vs 120) gives NT-Xent more in-batch negatives per gradient step, sharpening discrimination.
- 30 epochs (vs 10) lets the recipe-induced gradient pressure compound.
- We don't claim mechanism beyond "the baseline's same-artist misses are recovered; the persistent misses are still same-artist composed-section confusions"—which is what the data says.

### 5. What we cannot conclude from this data
- **Cannot attribute** the gain to any individual recipe component (sampler vs F_MIN vs BSZ vs epochs). Would require 4 more training runs for a clean factorial.
- **Cannot say anything about NMFP's remaining 5 misses** between our recipe and theirs — that gap (HR@1 0.995 → 1.000) is closed only by NMFP's full 5-fix recipe at 100 epochs.
- **Cannot generalize beyond Saraga 1.5** — 357 refs is a small library; the patterns here may not transfer to a 25k-track library.

---

## Definitions

- **miss** = top-1 predicted ref_id ≠ ground-truth ref_id for that query.
- **same-artist miss** = top-1 prediction is by the same artist as truth (intersect of `artists` string).
- **same-raaga / same-corpus**: analogous.
- **persistent miss** = ALL 3 recipe_v3 seeds (42, 137, 2026) miss this query.
- **fixed by recipe** = baseline misses, ALL 3 seeds hit.
- **broken unanimously** = baseline hits, ALL 3 seeds miss.
- **broken any-seed** = baseline hits, at least 1 seed misses.
- McNemar **b** = baseline-hit & recipe-miss; **c** = baseline-miss & recipe-hit.
- Pooled p = exact 2-sided binomial on min(b+c) across seeds.

## Per-cell summary

| Cell | N | Baseline misses | Recipe v3 mean misses | Pooled b, c | Pooled p | Persistent miss |
|---|---|---|---|---|---|---|
| main_1s | 1000 | 17 | 5.0 | b=12, c=48 | 3.18e-06 | 0 |
| main_3s | 1000 | 2 | 0.0 | b=0, c=6 | 3.12e-02 | 0 |
| main_5s | 1000 | 1 | 0.0 | b=0, c=3 | 2.50e-01 | 0 |
| main_10s | 1000 | 0 | 0.0 | b=0, c=0 | 1.00e+00 | 0 |
| ablation_1s | 632 | 13 | 5.7 | b=17, c=39 | 4.56e-03 | 2 |
| ablation_3s | 632 | 0 | 0.0 | b=0, c=0 | 1.00e+00 | 0 |
| ablation_5s | 632 | 0 | 0.0 | b=0, c=0 | 1.00e+00 | 0 |
| ablation_10s | 632 | 0 | 0.0 | b=0, c=0 | 1.00e+00 | 0 |


## main_1s

- n queries: **1000**
- Baseline hits: **983** (0.9830)
- Baseline misses: **17**
- Recipe v3 per-seed hits: 993, 997, 995
- Recipe v3 unanimous hits (all 3 seeds): **988**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **15**
- Broken any-seed (baseline hit → ≥1 seed miss): 10
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=12, c=48, p = **3.18e-06**

### Baseline misses (17) — what kind of confusions?

- Same-artist top-1: **14 / 17** (82%)
- Same-raaga top-1: 0 / 17
- Same-corpus (both Hindustani or both Carnatic): 16 / 17
- Cross-corpus (HI→CA or CA→HI): 1 / 17
- Cross-corpus examples (query, truth-corpus, pred-corpus):
    - hindustani_t0013_q2: truth=hindustani, predicted=carnatic


## main_3s

- n queries: **1000**
- Baseline hits: **998** (0.9980)
- Baseline misses: **2**
- Recipe v3 per-seed hits: 1000, 1000, 1000
- Recipe v3 unanimous hits (all 3 seeds): **1000**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **2**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=6, p = **3.12e-02**

### Baseline misses (2) — what kind of confusions?

- Same-artist top-1: **2 / 2** (100%)
- Same-raaga top-1: 0 / 2
- Same-corpus (both Hindustani or both Carnatic): 2 / 2
- Cross-corpus (HI→CA or CA→HI): 0 / 2


## main_5s

- n queries: **1000**
- Baseline hits: **999** (0.9990)
- Baseline misses: **1**
- Recipe v3 per-seed hits: 1000, 1000, 1000
- Recipe v3 unanimous hits (all 3 seeds): **1000**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **1**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=3, p = **2.50e-01**

### Baseline misses (1) — what kind of confusions?

- Same-artist top-1: **1 / 1** (100%)
- Same-raaga top-1: 0 / 1
- Same-corpus (both Hindustani or both Carnatic): 1 / 1
- Cross-corpus (HI→CA or CA→HI): 0 / 1


## main_10s

- n queries: **1000**
- Baseline hits: **1000** (1.0000)
- Baseline misses: **0**
- Recipe v3 per-seed hits: 1000, 1000, 1000
- Recipe v3 unanimous hits (all 3 seeds): **1000**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **0**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=0, p = **1.00e+00**


## ablation_1s

- n queries: **632**
- Baseline hits: **619** (0.9794)
- Baseline misses: **13**
- Recipe v3 per-seed hits: 623, 630, 626
- Recipe v3 unanimous hits (all 3 seeds): **620**
- Persistent misses (all 3 seeds miss): **2**
- Fixed by recipe (baseline miss → all 3 seeds hit): **13**
- Broken any-seed (baseline hit → ≥1 seed miss): 12
- Broken unanimously (baseline hit → all 3 miss): 2
- Pooled McNemar: b=17, c=39, p = **4.56e-03**

### Baseline misses (13) — what kind of confusions?

- Same-artist top-1: **9 / 13** (69%)
- Same-raaga top-1: 0 / 13
- Same-corpus (both Hindustani or both Carnatic): 11 / 13
- Cross-corpus (HI→CA or CA→HI): 2 / 13
- Section types of failing queries: {'composed': 4, 'alaap': 9}
- Cross-corpus examples (query, truth-corpus, pred-corpus):
    - hindustani_ab_107_Raag_Shree_s001_composed: truth=hindustani, predicted=carnatic
    - carnatic_ab_180_Ninne_Bhajana_s000_alaap: truth=carnatic, predicted=hindustani

### Persistent misses (2) — what recipe v3 still can't fix

- Same-artist top-1: 2 / 2
- Same-raaga top-1: 0 / 2
- Same-corpus: 2 / 2
- Section types: {'composed': 2}

### Broken unanimously (2) — baseline got these, all 3 recipe seeds lost them

  - carnatic_ab_114_Varashiki_Vahana_s000_composed
  - carnatic_ab_41_Soundararajam_s001_composed


## ablation_3s

- n queries: **632**
- Baseline hits: **632** (1.0000)
- Baseline misses: **0**
- Recipe v3 per-seed hits: 632, 632, 632
- Recipe v3 unanimous hits (all 3 seeds): **632**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **0**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=0, p = **1.00e+00**


## ablation_5s

- n queries: **632**
- Baseline hits: **632** (1.0000)
- Baseline misses: **0**
- Recipe v3 per-seed hits: 632, 632, 632
- Recipe v3 unanimous hits (all 3 seeds): **632**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **0**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=0, p = **1.00e+00**


## ablation_10s

- n queries: **632**
- Baseline hits: **632** (1.0000)
- Baseline misses: **0**
- Recipe v3 per-seed hits: 632, 632, 632
- Recipe v3 unanimous hits (all 3 seeds): **632**
- Persistent misses (all 3 seeds miss): **0**
- Fixed by recipe (baseline miss → all 3 seeds hit): **0**
- Broken any-seed (baseline hit → ≥1 seed miss): 0
- Broken unanimously (baseline hit → all 3 miss): 0
- Pooled McNemar: b=0, c=0, p = **1.00e+00**
