---
license: cc-by-nc-sa-4.0
license_link: LICENSE
language:
- hi
- sa
- ta
- te
- kn
tags:
- audio-fingerprinting
- music-information-retrieval
- indian-classical-music
- hindustani
- carnatic
- saraga
- benchmark
- nafp
- nmfp
- audio
task_categories:
- audio-classification
size_categories:
- 1K<n<10K
pretty_name: Audio Fingerprinting Benchmark on Indian Classical Music
configs:
- config_name: queries
  data_files:
  - split: test
    path: data/queries/**
- config_name: queries_ablation
  data_files:
  - split: test
    path: data/queries_ablation/**
- config_name: refs
  data_files:
  - split: library
    path: data/refs.parquet
- config_name: inspection_tracks
  data_files: [{split: library, path: data/inspection/tracks.parquet}]
- config_name: inspection_sections
  data_files: [{split: library, path: data/inspection/sections.parquet}]
- config_name: inspection_works
  data_files: [{split: library, path: data/inspection/works.parquet}]
- config_name: inspection_leakage_pairs
  data_files: [{split: library, path: data/inspection/leakage_pairs.parquet}]
---

# Audio Fingerprinting Benchmark on Indian Classical Music

Reproducible benchmark of **five audio-fingerprinting systems** on the
[Saraga 1.5 corpus](https://zenodo.org/records/4301737) (Hindustani + Carnatic),
plus a **pre-registered training-recipe improvement** to the NAFP baseline that
achieves Bonferroni-significant gains on 1-second queries.

**v0.6 (2026-05-14)** — see [Changelog](#changelog) for what's new.

---

## TL;DR

- **5 systems benchmarked**: Olaf, Dejavu, Panako (classical hash-based), NAFP
  (Chang et al. ICASSP 2021), NMFP (Araz et al. ISMIR 2025)
- **357 reference tracks** (108 Hindustani + 249 Carnatic) from Saraga 1.5
- **1,632 queries × 4 lengths** (1 s / 3 s / 5 s / 10 s) = **6,528 evaluation cells**
  per system
- **Pre-registered improvement** (recipe v3): NAFP HR@1 on 1-second queries
  improves from 0.983 → 0.995 mean across 3 seeds (pooled McNemar **p = 3.18 × 10⁻⁶**)
- **Two pre-registered negative results** reported with the same rigor
  (Intervention 2: per-artist mean subtraction; hubness post-processing)
- **Sample-accurate ground truth**: each query stores exact `offset_sec` in its
  source reference recording — alignment error measurable to sub-50 ms
- **Source MP3s not redistributed**: rebuild library by fetching Saraga 1.5
  directly from Zenodo

---

## Headline result (HR@1 across 5 systems × 4 query lengths)

### Main set (1,000 queries, no section-alignment constraint)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| Olaf | 0.492 | 0.955 | 0.993 | 0.998 |
| Dejavu | 0.745 | 0.969 | 0.994 | 1.000 |
| Panako | 0.000 | 0.000 | 0.922 | 0.997 |
| **NAFP** (Chang et al. 2021, our baseline) | **0.983** | 0.998 | 0.999 | 1.000 |
| **Recipe v3** (3-seed mean, this work) | **0.995** | 1.000 | 1.000 | 1.000 |
| NMFP-ckpt-100 (Araz et al. 2025, ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### Ablation set (632 section-aligned queries)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| Olaf | 0.494 | 0.907 | 0.987 | 1.000 |
| Dejavu | 0.764 | 0.978 | 0.998 | 1.000 |
| Panako | 0.000 | 0.000 | 0.929 | 0.994 |
| **NAFP** (baseline) | **0.979** | 1.000 | 1.000 | 1.000 |
| **Recipe v3** (3-seed mean) | **0.991** | 1.000 | 1.000 | 1.000 |
| NMFP-ckpt-100 (ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### Statistical significance (pooled across 3 seeds)

| Cell | Baseline | Recipe v3 (mean) | Pooled b, c | Pooled McNemar p |
|---|---|---|---|---|
| **main 1 s** | 0.983 (17 miss) | **0.995 (5 miss)** | b=12, c=48 | **3.18 × 10⁻⁶** ✓✓ |
| main 3 s | 0.998 | 1.000 | b=0, c=6 | 0.031 |
| main 5 s | 0.999 | 1.000 | b=0, c=3 | 0.250 |
| main 10 s | 1.000 | 1.000 | — | — |
| **ablation 1 s** | 0.979 (13 miss) | **0.991 (6 miss)** | b=17, c=39 | **0.0046** ✓✓ |
| ablation 3 s | 1.000 | 1.000 | — | — |
| ablation 5 s | 1.000 | 1.000 | — | — |
| ablation 10 s | 1.000 | 1.000 | — | — |

✓✓ = Bonferroni-significant at α / 8 = 0.00625.
No cell regresses on average. Recipe v3 closes the gap to the NMFP ceiling
substantially at ~10 % of NMFP's training compute.

---

## What's in this dataset

```
data/
├── queries/                         # AudioFolder
│   ├── metadata.parquet             # 1 000 main queries × ground-truth offsets
│   ├── hindustani/*.wav             # 500
│   └── carnatic/*.wav               # 500
├── queries_ablation/
│   ├── metadata.parquet             # 632 section-aligned queries
│   ├── hindustani/*.wav             # 207
│   └── carnatic/*.wav               # 425
├── refs.parquet                     # 357 ref tracks; metadata only (NO audio paths)
├── results/                         # Per (system × split × length) parquets + scores.json
│   ├── {system}_{split}_{length}.parquet           # ranked top-K candidates per query
│   ├── {system}_{split}_{length}.scores.json       # HR@k + Wilson CI + alignment error
│   ├── recipe_v3_seed{42,137,2026}_{split}_{length}.parquet
│   ├── recipe_v3_seed{42,137,2026}_{split}_{length}.scores.json
│   ├── recipe_v3_pooled_mcnemar.csv                # primary endpoint (pooled across seeds)
│   ├── PROTOCOL_recipe_v3.md                       # pre-registration (locked before training)
│   ├── PROTOCOL_intervention2.md                   # pre-registered NEGATIVE result
│   └── RESULTS_recipe_v3.md                        # full writeup
├── inspection/                      # Library audit tables
│   ├── tracks.parquet               # 357 rows: ref_id, corpus, raagas, taalas, artists, works, work_mbids
│   ├── sections.parquet             # ~2 000 rows: ref_id, start_sec, end_sec, section_type
│   ├── works.parquet                # unique work-MBIDs
│   └── leakage_pairs.parquet        # 130 (main_query → other_ref) pairs sharing a work-MBID
└── configs/                         # Seeded test-set generation configs (reproducibility)
    ├── hindustani_main.json
    ├── hindustani_ablation.json
    ├── carnatic_main.json
    └── carnatic_ablation.json
```

`{system}` ∈ `{olaf, dejavu, panako, nafp, nmfp}`. `{split}` ∈ `{main, ablation}`. `{length}` ∈ `{1s, 3s, 5s, 10s}`.

---

## How to load

```python
from datasets import load_dataset

# Audio queries with sample-accurate offsets
queries = load_dataset("Tachyeon/audio-fingerprint-indian-bench",
                       "queries", split="test")
# queries[0] → {'audio': {...}, 'query_id': ..., 'ref_id': ..., 'offset_sec': ...}

# Reference-track metadata
refs = load_dataset("Tachyeon/audio-fingerprint-indian-bench",
                    "refs", split="library")

# Per-system results — load any specific (system × split × length) parquet
import pandas as pd
from huggingface_hub import hf_hub_download

scores_path = hf_hub_download(
    repo_id="Tachyeon/audio-fingerprint-indian-bench",
    filename="data/results/recipe_v3_seed42_main_1s.scores.json",
    repo_type="dataset",
)
import json
print(json.load(open(scores_path))["hr@1"])  # → 0.993
```

---

## Methodology

### Query construction
- 10-second clips at 16 kHz mono, cut at random offsets from Saraga ref tracks
- **Sample-accurate**: each query stores `(ref_id, offset_sec, seed)` — ground
  truth alignment is bit-exact
- **Main set (1 000)**: no section-alignment constraint
- **Ablation set (632)**: queries align to Saraga's `section_annotation`
  metadata; `section_type` ∈ `{alaap, composed, tani}` allows per-section breakdowns
- Shorter (1/3/5 s) queries are first-N truncations of the 10-second cuts —
  *not* random re-cuts. Documented as a limitation; see [Limitations](#limitations).

### Reference library
- 357 Saraga 1.5 concert recordings (no other audio added)
- No distractors / no FMA mix — clean retrieval benchmark
- 130 main queries share a MusicBrainz work-MBID with a different recording in
  the library (composition-twin leakage). Tracked explicitly via
  `inspection/leakage_pairs.parquet`; we report HR@1 separately for
  with-twin / no-twin subsets in `scores.json`.

### Scoring
- **HR@k**: fraction of queries where the truth `ref_id` is among the top-k
  predicted refs. Reported for k ∈ {1, 5, 10} with Wilson 95% CI.
- **MRR@k**: mean reciprocal rank, capped at k.
- **top1_near**: HR@1 at coarse temporal accuracy (within ±0.5 s of truth);
  follows NAFP-paper convention.
- **Alignment error**: per-query offset error (median, p95, max).

### Statistical comparison
- **McNemar exact test** on paired binary (hit/miss) outcomes per query.
- **3-seed pooled McNemar** for the recipe v3 primary endpoint (3,000 paired
  observations per cell).
- **Bonferroni correction** at α / 8 = 0.00625 across the 8 (split × length) cells.

---

## Systems benchmarked

| System | Type | Training | Reference |
|---|---|---|---|
| **Olaf** | Classical (constellation hash) | None (rule-based) | https://github.com/JorenSix/Olaf |
| **Dejavu** | Classical (peak pairs, PostgreSQL) | None | https://github.com/worldveil/dejavu |
| **Panako** | Classical (CQT triplet hash) | None | http://panako.be |
| **NAFP** | Neural CNN + NT-Xent contrastive | 10 epochs on FMA-medium 10k_icassp | [Chang et al. ICASSP 2021](https://arxiv.org/abs/2010.11910) |
| **NMFP** | Neural CNN (same arch as NAFP) | 100 epochs on FMA-medium with 5 recipe fixes | [Araz et al. ISMIR 2025](https://arxiv.org/abs/2506.22661) |

NMFP weights are pre-trained by Araz et al. (Zenodo 15719945, GPLv3 / AGPLv3 —
viral); we use them only to establish the ceiling and do not redistribute.

---

## Recipe v3 — pre-registered training-recipe improvement (this work)

**Hypothesis** (locked at `data/results/PROTOCOL_recipe_v3.md` before training):
two of NMFP's published recipe fixes, combined with a 3× larger batch and 3×
longer schedule, will improve NAFP's same-artist failure mode on 1-second
queries with Bonferroni-significant gain across 8 cells × 3 seeds.

**Recipe**: F_MIN: 300 → 160 Hz; one-anchor-per-track sampler (re-sampled per
epoch); BSZ: 120 → 320; MAX_EPOCH: 10 → 30; NT-Xent τ=0.05 preserved; Adam,
cosine LR. Trained from scratch on FMA-medium (same training corpus as baseline).

**Result**: primary endpoint cleared — main_1s pooled p = 3.18 × 10⁻⁶ (clears
Bonferroni α / 8 by ~1900×). ablation_1s pooled p = 0.0046 (Bonferroni-sig).
No cell regresses on average.

**What we did NOT do** (honest disclosure): we did not apply NMFP's other 3
recipe fixes (full-IR augmentation, 1-sec acoustic history, triplet loss with
semi-hard mining) because they require dataloader surgery beyond our budget;
adding them is the path to closing the residual ~0.005 gap to NMFP's 1.000
ceiling.

---

## Pre-registered negative results

We use a pre-registration discipline: every intervention's hypothesis, primary
endpoint, and falsification rule are committed before training data is
collected. Two interventions tested with this protocol returned negative:

### Intervention 2: per-artist mean subtraction at inference
- **Hypothesis**: subtract a learned per-artist centroid from each NAFP ref
  embedding to reduce same-artist top-1 confusions
- **α sweep** {0, 0.05, …, 0.30} + 3 controls (α=0 sanity, shuffled-centroid,
  isotropic-centroid)
- **Result**: REJECTED. Isotropic-centroid control matched V1 in 4/4 main cells
  → mechanism falsified. The same-artist confusion is not addressable by
  inference-time centroid correction.
- Protocol: `data/results/PROTOCOL_intervention2.md`

### Hubness post-processing (Inverted Softmax + CSLS)
- **Hypothesis**: NAFP's same-artist failure mode is a generic hubness problem
  fixable at inference time via Smith et al. 2017 / Conneau et al. 2018 style
  re-scoring
- **Result**: REJECTED. Both methods reproduced baseline HR@1 exactly across
  all 8 cells (no gain, no regression). The failure is encoder-level, not
  embedding-geometry.

Both negatives strengthen the recipe v3 positive result: we ruled out
inference-only fixes, isolating training-recipe modifications as the actual
lever.

---

## Reproducibility

```bash
# 1. Fetch dataset metadata + queries + results parquets
from datasets import load_dataset
queries = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "queries", split="test")

# 2. Fetch Saraga 1.5 source MP3s (NOT in this dataset; required for reference library)
# via mirdata:
#   pip install mirdata
#   import mirdata
#   mirdata.initialize("saraga_hindustani", data_home="~/saraga_hindustani").download()
#   mirdata.initialize("saraga_carnatic",   data_home="~/saraga_carnatic").download()

# 3. Code + system runners + recipe v3 pipeline:
#   https://github.com/ipritamdash/afp-indian-classical (currently private; ask author)

# 4. NMFP-ckpt-100 weights (for ceiling reference, optional):
#   Zenodo: https://zenodo.org/records/15719945
#   License: GPLv3 / AGPLv3 (viral); review before bundling into derivative work
```

Per-query result parquets are deterministic given fixed (system, seed, audio).
Recipe v3 random_seed = 42, 137, 2026.

---

## Limitations

- **Library is small (357 refs)** — HR@1 saturates for 5/10 s queries on the
  baseline. The hardest cell (1 s, same-artist) is where the improvement lives.
- **No FMA distractors / no expanded gallery** — a more realistic deployment
  benchmark would mix in 10k+ Western tracks. Out of scope here.
- **Shorter queries are first-N truncations**, not random re-cuts — known
  methodological gap from the Chang 2021 protocol; documented but not fixed.
- **Recipe v3 trades epoch parity for convergence** — trained 30 ep vs
  baseline's 10 ep. The improvement is recipe + 3× longer training combined.
  Individual ablations not run.
- **3 seeds is the minimum** for stable pooled-McNemar; 5+ seeds preferable in
  follow-up.
- **NMFP-ckpt-100 remains the ceiling** at HR@1 = 1.000 across all 8 cells. We
  reach ~95% of that ceiling at ~10% of their training compute, but do not
  beat them.
- **Tani ablation cell N = 11** (Hindustani: 0; Carnatic only) — per-section
  HR@1 claims for Tani are underpowered.

---

## Licensing

| Component | License |
|---|---|
| Reference-track metadata, query metadata, inspection tables | CC-BY-NC-SA 4.0 (inherits Saraga) |
| Query WAVs (derived from Saraga source) | CC-BY-NC-SA 4.0 |
| Per-system result parquets, scores.json | CC-BY-NC-SA 4.0 (covers derivative work definition) |
| Source MP3s | NOT distributed; fetch from Zenodo (CC-BY-NC-SA 4.0) |
| NMFP teacher weights | NOT distributed; fetch from Zenodo 15719945 (GPLv3 / AGPLv3) |
| Code (separate GitHub repo) | MIT (sharply scoped to repo code, see github.com/ipritamdash/afp-indian-classical) |

See `LICENSE`, `LICENSE-CODE`, and `ATTRIBUTION.md` in this repo for full text.

---

## Citation

```bibtex
@misc{afp-indian-classical-2026,
  author       = {Pritam Kumar},
  title        = {Audio Fingerprinting Benchmark on Indian Classical Music},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench}},
  note         = {Includes pre-registered training-recipe improvement to NAFP (Chang et al. 2021).},
}
```

When citing this benchmark, please also cite the upstream papers:

```bibtex
@dataset{srinivasamurthy2021saraga,
  title     = {{Saraga}: Open Datasets for Research on {I}ndian Art Music},
  author    = {Srinivasamurthy, Ajay and Gulati, Sankalp and Repetto, Rafael Caro and Serra, Xavier},
  year      = {2021}, version = {1.5},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4301737},
}
@inproceedings{chang2021nafp,
  title     = {{Neural Audio Fingerprint} for High-specific Audio Retrieval based on Contrastive Learning},
  author    = {Chang, Sungkyun and Lee, Donmoon and Park, Jeongsoo and Lim, Hyungui and Lee, Kyogu and Ko, Karam and Han, Yoonchang},
  booktitle = {ICASSP}, year = {2021}, doi = {10.1109/ICASSP39728.2021.9414083},
}
@inproceedings{araz2025nmfp,
  title     = {Enhancing Neural Audio Fingerprint Robustness to Real-World Conditions},
  author    = {Araz, R. O. and Cortès-Sebastià, G. and Molina, E. and Serra, X. and Serra, J. and Mitsufuji, Y. and Bogdanov, D.},
  booktitle = {ISMIR}, year = {2025}, eprint = {arXiv:2506.22661},
}
```

---

## Changelog

- **v0.6** (2026-05-14): Major release.
  - **Added NMFP-ckpt-100** (Araz et al. ISMIR 2025) as system #5 — establishes
    HR@1 = 1.000 ceiling across all 8 cells
  - **Added Recipe v3** training-recipe improvement (3 seeds × 30 epochs ×
    BSZ=320); pre-registered Bonferroni-significant on main_1s + ablation_1s
  - **Added pre-registered negative results**: Intervention 2 (per-artist mean
    subtraction) and hubness post-processing (InvSoftmax + CSLS)
  - All 4 query lengths now exposed per (system × split); v0.5 only had 10 s
  - Added pooled-McNemar CSV + protocol markdowns for full transparency
  - README rewritten for clarity
- **v0.5** (2026-05-12): Post-mortem audit fixes. `has_twin_in_library`
  recomputed via work_mbids only (was works-text); Dejavu `ref_stop` fixed;
  `score.py` `no_match` denominator fixed; NAFP `np.argsort` made stable.
  Headline HR@1 unchanged; twin/no-twin breakdown shifted (165 → 130 confirmed
  twins).
- **v0.4** (2026-05-12): NAFP added as 4th system.
- **v0.3** (2026-05-12): Refreshed ablation result parquets to match v0.2
  manifest. Length-curve added.
- **v0.2** (2026-05-12): F5 fix (NFKD ablation bucketing); 624 → 632 ablation
  queries.
- **v0.1** (2026-05-12): Initial release with 3 classical systems on 624
  ablation queries.

---

## Attribution

See `ATTRIBUTION.md` for full credits to Saraga / NAFP / NMFP / FMA upstream authors.
