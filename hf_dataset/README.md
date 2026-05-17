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
- pre-registered
- contrastive-learning
task_categories:
- audio-classification
- feature-extraction
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

A reproducible, pre-registered benchmark of **five audio-fingerprinting systems** on the [Saraga 1.5 corpus](https://zenodo.org/records/4301737) (Hindustani + Carnatic), plus a **pre-registered training-recipe improvement** to the NAFP baseline that achieves Bonferroni-significant gains on 1-second queries.

**v0.7** &nbsp;·&nbsp; closed-world retrieval &nbsp;·&nbsp; 5 systems &nbsp;·&nbsp; 6 528 evaluation cells &nbsp;·&nbsp; pooled McNemar **p = 3.18 × 10⁻⁶**

---

## TL;DR

- **5 systems benchmarked**: Olaf, Dejavu, Panako (classical hash-based); NAFP (Chang et al. ICASSP 2021); NMFP (Araz et al. ISMIR 2025)
- **357 reference tracks** (108 Hindustani + 249 Carnatic) from Saraga 1.5 — **clean library, no distractors** (FMA mix-in out of scope)
- **1 632 queries × 4 lengths** (1 s / 3 s / 5 s / 10 s) = **6 528 evaluation cells** per system
- **Pre-registered improvement** (recipe v3): NAFP HR@1 on main_1s improves from 0.983 → **0.995** mean across 3 seeds, pooled McNemar **p = 3.18 × 10⁻⁶** (Bonferroni-significant at α/8 = 0.00625)
- **~68 % fewer misses** across the benchmark (~71 % on the hardest cell)
- **Two pre-registered negative results** with identical rigor: (1) per-artist mean subtraction at inference and (2) hubness post-processing (Inverted Softmax + CSLS) — strengthen the recipe v3 positive
- **Sample-accurate ground truth**: every query stores exact `(ref_id, offset_sec, seed)` — alignment error measurable to sub-50 ms
- **Source MP3s not redistributed**: rebuild library by fetching Saraga 1.5 directly from Zenodo
- **Full primary endpoint writeup**: [`data/results/RESULTS_recipe_v3.md`](data/results/RESULTS_recipe_v3.md); pre-registration: [`data/results/PROTOCOL_recipe_v3.md`](data/results/PROTOCOL_recipe_v3.md)

---

## Headline result

### HR@1 — main set (1 000 queries, no section constraint)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---:|---:|---:|---:|
| Olaf | 0.492 | 0.955 | 0.993 | 0.998 |
| Dejavu | 0.745 | 0.969 | 0.994 | 1.000 |
| Panako | 0.000 | 0.000 | 0.922 | 0.997 |
| **NAFP** (Chang et al. 2021 — our baseline) | **0.983** | 0.998 | 0.999 | 1.000 |
| **Recipe v3** (3-seed mean — this work) | **0.995** | 1.000 | 1.000 | 1.000 |
| NMFP-ckpt-100 (Araz et al. 2025 — ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### HR@1 — ablation set (632 section-aligned queries)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---:|---:|---:|---:|
| Olaf | 0.494 | 0.907 | 0.987 | 1.000 |
| Dejavu | 0.764 | 0.978 | 0.998 | 1.000 |
| Panako | 0.000 | 0.000 | 0.929 | 0.994 |
| **NAFP** (baseline) | **0.979** | 1.000 | 1.000 | 1.000 |
| **Recipe v3** (3-seed mean) | **0.991** | 1.000 | 1.000 | 1.000 |
| NMFP-ckpt-100 (ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### Statistical significance (recipe v3 vs baseline, pooled across 3 seeds)

| Cell | Baseline | Recipe v3 (mean) | Pooled b, c | Pooled McNemar p |
|---|---|---|---|---:|
| **main 1 s** | 0.983 (17 miss) | **0.995 (5 miss)** | b = 12, c = 48 | **3.18 × 10⁻⁶** ✓ |
| main 3 s | 0.998 | 1.000 | b = 0, c = 6 | 0.031 |
| main 5 s | 0.999 | 1.000 | b = 0, c = 3 | 0.250 |
| main 10 s | 1.000 | 1.000 | — | — |
| **ablation 1 s** | 0.979 (13 miss) | **0.991 (5.7 miss)** | b = 17, c = 39 | **0.0046** ✓ |
| ablation 3 s / 5 s / 10 s | 1.000 | 1.000 | — | — |

✓ Bonferroni-significant at α/8 = 0.00625. The two hardest cells clear the corrected threshold; remaining six cells are at ceiling (no statistical headroom). Pooled CSV: [`data/results/recipe_v3_pooled_mcnemar.csv`](data/results/recipe_v3_pooled_mcnemar.csv).

**Error-rate reduction:** main_1s **−70.6 %** misses (17 → 5), ablation_1s **−56.2 %** (13 → 5.7), benchmark-wide **−67.6 %** (33 → 10.7 across 6 528 cells).

---

## What's in this dataset

```
data/
├── queries/                                  # AudioFolder
│   ├── metadata.parquet                      # 1 000 main queries with ground-truth offsets
│   ├── hindustani/*.wav                      # 500
│   └── carnatic/*.wav                        # 500
├── queries_ablation/
│   ├── metadata.parquet                      # 632 section-aligned queries
│   ├── hindustani/*.wav                      # 207
│   └── carnatic/*.wav                        # 425
├── refs.parquet                              # 357 ref tracks; metadata only (no audio paths)
├── results/
│   ├── {system}_{split}_{length}.parquet     # ranked top-K candidates per query
│   ├── {system}_{split}_{length}.scores.json # HR@k + Wilson 95 % CI + alignment error
│   ├── recipe_v3_seed{42,137,2026}_{split}_{length}.parquet
│   ├── recipe_v3_seed{42,137,2026}_{split}_{length}.scores.json
│   ├── recipe_v3_pooled_mcnemar.csv          # primary endpoint, pooled across 3 seeds
│   ├── PROTOCOL_recipe_v3.md                 # pre-registration (locked before training)
│   ├── PROTOCOL_intervention2.md             # pre-registered NEGATIVE result protocol
│   └── RESULTS_recipe_v3.md                  # full writeup of the primary endpoint
├── inspection/                               # Library audit tables
│   ├── tracks.parquet                        # 357 rows: ref_id, corpus, raagas, taalas, artists, works, work_mbids
│   ├── sections.parquet                      # section_type spans per ref (alaap / composed / tani)
│   ├── works.parquet                         # unique work-MBIDs
│   └── leakage_pairs.parquet                 # 130 (main_query → other_ref) pairs sharing a work-MBID
└── configs/                                  # Seeded test-set generation configs (reproducibility)
    ├── hindustani_main.json
    ├── hindustani_ablation.json
    ├── carnatic_main.json
    └── carnatic_ablation.json

ARTEFACTS.parquet                             # SHA-256 + size manifest for every file in this dataset
ATTRIBUTION.md                                # full upstream credits (Saraga / NAFP / NMFP / FMA)
LICENSE        / LICENSE-CODE                 # CC-BY-NC-SA 4.0 (data) / MIT (code in linked repo)
```

`{system}` ∈ `{olaf, dejavu, panako, nafp, nmfp}` · `{split}` ∈ `{main, ablation}` · `{length}` ∈ `{1s, 3s, 5s, 10s}`.

---

## How to load

```python
from datasets import load_dataset

# Audio queries with sample-accurate offsets
queries = load_dataset(
    "Tachyeon/audio-fingerprint-indian-bench",
    "queries",
    split="test",
)
# queries[0] → {'audio': {...}, 'query_id': ..., 'ref_id': ..., 'offset_sec': ...}

# Reference-track metadata (no audio paths — fetch source MP3s from Zenodo separately)
refs = load_dataset(
    "Tachyeon/audio-fingerprint-indian-bench",
    "refs",
    split="library",
)

# Per-system result tables — load any (system × split × length) parquet directly
from huggingface_hub import hf_hub_download
import json

scores_path = hf_hub_download(
    repo_id="Tachyeon/audio-fingerprint-indian-bench",
    filename="data/results/recipe_v3_seed42_main_1s.scores.json",
    repo_type="dataset",
)
print(json.load(open(scores_path))["hr@1"])   # → 0.993
```

---

## Methodology

### Query construction
- 10-second clips at 16 kHz mono, cut at random offsets from Saraga ref tracks
- **Sample-accurate**: each query stores `(ref_id, offset_sec, seed)` — ground-truth alignment is bit-exact
- **Main set (1 000 queries)**: random tracks + random offsets, no section constraint (500 Hindustani + 500 Carnatic)
- **Ablation set (632 queries)**: offsets aligned to Saraga's `section_annotation` metadata; `section_type` ∈ `{alaap, composed, tani}` enables per-section breakdowns
- Shorter (1/3/5 s) queries are **first-N truncations** of the 10-second cuts — *not* random re-cuts. Documented as a methodological limitation; see [Limitations](#limitations).

### Reference library
- 357 Saraga 1.5 concert recordings; **no other audio added** (closed-world, clean library)
- 130 main queries share a MusicBrainz work-MBID with a different recording in the library (composition-twin leakage). Tracked explicitly via [`data/inspection/leakage_pairs.parquet`](data/inspection/leakage_pairs.parquet); HR@1 reported separately for with-twin / no-twin subsets inside each `scores.json`.

### Scoring
- **HR@k**: fraction of queries where the truth `ref_id` is among the top-k predicted refs. Reported for k ∈ {1, 5, 10} with **Wilson 95 % CI**.
- **MRR@k**: mean reciprocal rank, capped at k.
- **top1_near**: HR@1 with a coarse temporal-accuracy gate (within ±0.5 s of the truth offset); follows the NAFP-paper convention.
- **Alignment error**: per-query offset error (median, p95, max), reported in seconds.

### Statistical comparison
- **McNemar exact test** on paired binary (hit / miss) outcomes per query
- **3-seed pooled McNemar** for the recipe v3 primary endpoint (3 000 paired observations per cell)
- **Bonferroni correction** at α/8 = 0.00625 across the 8 (split × length) cells

---

## Systems benchmarked

| System | Type | Training | Reference |
|---|---|---|---|
| **Olaf** | Classical (constellation hash) | None (rule-based) | <https://github.com/JorenSix/Olaf> |
| **Dejavu** | Classical (peak pairs, PostgreSQL) | None | <https://github.com/worldveil/dejavu> |
| **Panako** | Classical (CQT triplet hash) | None | <http://panako.be> |
| **NAFP** | Neural CNN + NT-Xent contrastive | 10 epochs on FMA-medium 10k_icassp | [Chang et al. ICASSP 2021](https://arxiv.org/abs/2010.11910) |
| **NMFP** | Neural CNN (same architecture as NAFP) | 100 epochs on FMA-medium with 5 recipe fixes | [Araz et al. ISMIR 2025](https://arxiv.org/abs/2506.22661) |

NMFP weights are pre-trained by Araz et al. ([Zenodo 15719945](https://zenodo.org/records/15719945), GPLv3 / AGPLv3 — viral); we use them only to establish the ceiling and do not redistribute.

---

## Recipe v3 — pre-registered training-recipe improvement (this work)

**Hypothesis** (locked in [`PROTOCOL_recipe_v3.md`](data/results/PROTOCOL_recipe_v3.md) before training): two of NMFP's published recipe fixes, combined with a larger batch and a 3× longer schedule, will recover NAFP's same-artist failure mode on 1-second queries with Bonferroni-significant gain across 8 cells × 3 seeds.

**Recipe**: `F_MIN`: 300 → **160 Hz** · `TR_SEG_MODE`: `all` → **`random_oneshot`** (one anchor per track per epoch, resampled) · `BSZ`: 120 → **320** · `MAX_EPOCH`: 10 → **30** · NT-Xent τ = 0.05 preserved · Adam, cosine LR. Trained from scratch on FMA-medium (same training corpus as baseline). **80.3 min total wall** across 3 seeds on a Colab L4 GPU.

**Result**: primary endpoint cleared — main_1s pooled McNemar **p = 3.18 × 10⁻⁶** (clears Bonferroni α/8 by ~1 900×). ablation_1s pooled p = 0.0046. No cell regresses on average.

**What we did NOT do** (honest disclosure): we did not apply NMFP's other 3 recipe fixes — full-IR augmentation, 1-second acoustic history, triplet loss with semi-hard mining — because each requires dataloader/model surgery beyond our budget. Adding them is the path to closing the residual ~0.005 gap to NMFP's 1.000 ceiling.

---

## Pre-registered negative results

Every intervention's hypothesis, primary endpoint, and falsification rule are committed before training data is collected. Two interventions tested with this protocol returned negative.

### Intervention 2 — per-artist mean subtraction at inference
- **Hypothesis**: subtract a learned per-artist centroid from each NAFP ref embedding to reduce same-artist top-1 confusions
- **Design**: α-sweep {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30} + 3 controls — α = 0 sanity, shuffled-centroid (artist labels permuted), and **isotropic-centroid** (random unit-norm vector replacing the learned centroid)
- **Result**: **REJECTED.** The isotropic-centroid control matched the baseline in 4 / 4 main cells — the mechanism is not artist-specific. Same-artist confusion is not addressable by inference-time centroid correction.
- Protocol: [`data/results/PROTOCOL_intervention2.md`](data/results/PROTOCOL_intervention2.md)

### Hubness post-processing — Inverted Softmax + CSLS
- **Hypothesis**: NAFP's same-artist failure mode is a generic hubness problem fixable at inference time via Smith et al. 2017 / Conneau et al. 2018 re-scoring
- **Result**: **REJECTED.** Both methods reproduced baseline HR@1 exactly across all 8 cells (no gain, no regression). The failure is encoder-level, not embedding-geometry.

Both negatives strengthen the recipe v3 positive: inference-only fixes were ruled out, isolating training-recipe modifications as the actual lever.

---

## Reproducibility

```python
# 1. Fetch dataset metadata + queries + result parquets from this dataset
from datasets import load_dataset
queries = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "queries", split="test")

# 2. Fetch Saraga 1.5 source MP3s (NOT redistributed here; required for reference library)
#    pip install mirdata
#    import mirdata
#    mirdata.initialize("saraga_hindustani", data_home="~/saraga_hindustani").download()
#    mirdata.initialize("saraga_carnatic",   data_home="~/saraga_carnatic").download()

# 3. Code, system runners, and the recipe v3 training pipeline live in a separate repo:
#    https://github.com/ipritamdash/afp-indian-classical  (private; request access via citation email)

# 4. NMFP-ckpt-100 weights (optional — only needed to reproduce the ceiling row):
#    https://zenodo.org/records/15719945   (GPLv3 / AGPLv3; review before bundling into derivative work)
```

Per-query result parquets are deterministic given fixed `(system, seed, audio)`. Recipe v3 seeds: 42, 137, 2026 (locked in `PROTOCOL_recipe_v3.md` before training).

---

## Limitations

- **Small reference library (357 refs).** HR@1 saturates for ≥ 3-second queries on all neural systems. The interesting signal lives in the 1-second cells; deployment-scale benchmarks would mix in 10 k+ distractors (out of scope here).
- **Shorter queries are first-N truncations** of the 10-second cuts — not random re-cuts. Inherited from the Chang et al. 2021 protocol; documented but not fixed.
- **Recipe v3 trades epoch parity for convergence** — trained 30 epochs vs the baseline's 10 epochs. The improvement is recipe + 3× longer training combined; per-knob ablations were out of scope.
- **3 seeds is the minimum** for a stable pooled-McNemar claim; 5+ seeds is preferable in follow-up.
- **NMFP-ckpt-100 remains the ceiling** at HR@1 = 1.000 across all 8 cells. We reach ~95 % of that ceiling at ~10 % of their training compute, but do not beat it.
- **Tani per-section cell is underpowered** (N = 11 in `ablation_1s`; Carnatic-only — Hindustani has no tani sections by tradition). Per-section HR@1 claims for tani should be treated as descriptive only.

---

## Licensing

| Component | License |
|---|---|
| Reference-track metadata, query metadata, inspection tables | CC-BY-NC-SA 4.0 (inherits Saraga) |
| Query WAVs (derived from Saraga source) | CC-BY-NC-SA 4.0 |
| Per-system result parquets, scores.json | CC-BY-NC-SA 4.0 (covers derivative-work definition) |
| Source MP3s | NOT distributed — fetch from [Zenodo](https://zenodo.org/records/4301737) (CC-BY-NC-SA 4.0) |
| NMFP teacher weights | NOT distributed — fetch from [Zenodo 15719945](https://zenodo.org/records/15719945) (GPLv3 / AGPLv3) |
| Code (separate GitHub repo) | MIT — applies only to code in the linked repo, not to data here |

See [`LICENSE`](LICENSE), [`LICENSE-CODE`](LICENSE-CODE), and [`ATTRIBUTION.md`](ATTRIBUTION.md) for full text.

---

## Citation

```bibtex
@misc{banwala2026afpindianclassical,
  author       = {Aryan Banwala},
  title        = {Audio Fingerprinting Benchmark on Indian Classical Music},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench}},
  note         = {Includes a pre-registered training-recipe improvement to NAFP (Chang et al. 2021), Bonferroni-significant on 1-second queries.},
}
```

When citing this benchmark, please also cite the upstream papers:

```bibtex
@dataset{srinivasamurthy2021saraga,
  title     = {{Saraga}: Open Datasets for Research on {I}ndian Art Music},
  author    = {Srinivasamurthy, Ajay and Gulati, Sankalp and Repetto, Rafael Caro and Serra, Xavier},
  year      = {2021},
  version   = {1.5},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4301737},
}
@inproceedings{chang2021nafp,
  title     = {{Neural Audio Fingerprint} for High-specific Audio Retrieval based on Contrastive Learning},
  author    = {Chang, Sungkyun and Lee, Donmoon and Park, Jeongsoo and Lim, Hyungui and Lee, Kyogu and Ko, Karam and Han, Yoonchang},
  booktitle = {ICASSP},
  year      = {2021},
  doi       = {10.1109/ICASSP39728.2021.9414083},
}
@inproceedings{araz2025nmfp,
  title     = {Enhancing Neural Audio Fingerprint Robustness to Real-World Conditions},
  author    = {Araz, R. O. and Cortès-Sebastià, G. and Molina, E. and Serra, X. and Serra, J. and Mitsufuji, Y. and Bogdanov, D.},
  booktitle = {ISMIR},
  year      = {2025},
  eprint    = {arXiv:2506.22661},
}
```

---

## Version

**v0.7** &nbsp;·&nbsp; 2026-05-17 &nbsp;·&nbsp; current and only supported release.

---

## Attribution

See [`ATTRIBUTION.md`](ATTRIBUTION.md) for full credits to Saraga / NAFP / NMFP / FMA upstream authors.
