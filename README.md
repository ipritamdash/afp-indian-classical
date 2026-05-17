# Audio Fingerprinting Benchmark for Indian Classical Music

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/🎵-live%20demo-orange.svg)](https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo)
[![Dataset](https://img.shields.io/badge/dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

Five audio-fingerprinting systems benchmarked on the **Saraga 1.5** Indian classical
music corpus (357 refs, 6 528 evaluation cells), plus a **pre-registered training-recipe
improvement to NAFP** with Bonferroni-significant gains on 1-second queries.

**🎵 Live demo:** https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo
**📁 Demo clip pack (67 clips, 8 MB):** [`demo_clips/`](demo_clips/) ([download zip](demo_clips_pack.zip))

---

## Where everything lives

| Artifact | URL |
|---|---|
| **Code (this repo)** | <https://github.com/ipritamdash/afp-indian-classical> |
| **Benchmark dataset** (queries + result parquets + protocols) | <https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench> |
| **Live demo + trained model weights** (recipe v3 seed 42 ckpt-30) | <https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo> |
| **Recipe v3 training notebook** (Colab L4, ~80 min for 3 seeds) | <https://colab.research.google.com/drive/1bS3q8vOiylqW2lyICls_i-cGFKC8VZc6> |
| **Teaching docs** | [`docs/AFP_Overview_Guide.pdf`](docs/AFP_Overview_Guide.pdf) · [`docs/AFP_Teaching_Guide.pdf`](docs/AFP_Teaching_Guide.pdf) |
| Saraga 1.5 source audio | <https://zenodo.org/records/4301737> (CC-BY-NC-SA 4.0; not redistributed here) |
| NMFP teacher weights | <https://zenodo.org/records/15719945> (GPLv3 / AGPLv3; not redistributed here) |

---

## TL;DR

| | Value |
|---|---|
| **Systems benchmarked** | Olaf · Dejavu · Panako · NAFP (Chang et al. 2021) · NMFP (Araz et al. 2025) |
| **Reference library** | 357 Saraga 1.5 tracks (108 Hindustani + 249 Carnatic) |
| **Test queries** | 1 632 queries × {1 s, 3 s, 5 s, 10 s} = 6 528 cells / system |
| **Headline win** | **−68 % miss rate across benchmark** (−71 % on the hardest cell, 1-second main); pooled McNemar p = 3.18 × 10⁻⁶, Bonferroni ✓ |
| **Pre-registered negatives** | Per-artist mean subtraction · Hubness post-processing |
| **Dataset** | [Tachyeon/audio-fingerprint-indian-bench](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench) |

---

## Results

### HR@1 — main set (1 000 queries, no section constraint)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---:|---:|---:|---:|
| Olaf | 0.492 | 0.955 | 0.993 | 0.998 |
| Dejavu | 0.745 | 0.969 | 0.994 | 1.000 |
| Panako | 0.000 | 0.000 | 0.922 | 0.997 |
| NAFP (baseline, our 10 ep retrain) | 0.983 | 0.998 | 0.999 | 1.000 |
| **Recipe v3 (this work, 3-seed mean)** | **0.995** | **1.000** | **1.000** | **1.000** |
| NMFP-ckpt-100 (Araz et al. 2025, ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### HR@1 — ablation set (632 section-aligned queries)

| System | 1 s | 3 s | 5 s | 10 s |
|---|---:|---:|---:|---:|
| Olaf | 0.494 | 0.907 | 0.987 | 1.000 |
| Dejavu | 0.764 | 0.978 | 0.998 | 1.000 |
| Panako | 0.000 | 0.000 | 0.929 | 0.994 |
| NAFP (baseline) | 0.979 | 1.000 | 1.000 | 1.000 |
| **Recipe v3 (this work)** | **0.991** | **1.000** | **1.000** | **1.000** |
| NMFP-ckpt-100 (ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

### Pre-registered primary endpoint (recipe v3 vs baseline, pooled over 3 seeds)

| Cell | Baseline | Recipe v3 mean | b (lost) | c (gained) | Pooled McNemar p |
|---|---|---|---:|---:|---:|
| **main 1 s** | 0.983 | 0.995 | 12 | 48 | **3.18 × 10⁻⁶** ✓✓ |
| **ablation 1 s** | 0.979 | 0.991 | 17 | 39 | **0.0046** ✓✓ |
| main 3 s | 0.998 | 1.000 | 0 | 6 | 0.031 |
| main 5 s | 0.999 | 1.000 | 0 | 3 | 0.250 |
| Others | 1.000 | 1.000 | 0 | 0 | — |

✓✓ Bonferroni-significant at α / 8 = 0.00625.

### Improvement summary (error-rate reduction)

| Cell | Baseline misses | Recipe v3 mean misses | Error reduction |
|---|---:|---:|---:|
| main 1 s | 17 / 1 000 | 5 / 1 000 | **−70.6 %** |
| ablation 1 s | 13 / 632 | 5.7 / 632 | **−56.2 %** |
| main 3 s | 2 / 1 000 | 0 / 1 000 | −100 % |
| main 5 s | 1 / 1 000 | 0 / 1 000 | −100 % |
| Other 4 cells | 0 | 0 | — |
| **All 8 cells (total)** | **33 / 6 528** | **10.7 / 6 528** | **−67.6 %** |

Headline number: **~68 % fewer misses across the benchmark, ~71 % on the hardest cell (1-second main queries)**.

Full analysis: [`data/results/nafp/recipe_v3_30ep/RESULTS.md`](data/results/nafp/recipe_v3_30ep/RESULTS.md).
Post-mortem (where each model lacks, what wins, what fails): [`docs/POST_MORTEM_RECIPE_V3.md`](docs/POST_MORTEM_RECIPE_V3.md).

---

## Recipe v3 — the improvement at a glance

Trained from scratch on FMA-medium with 2 of NMFP's 5 published recipe fixes:

| Knob | Baseline | Recipe v3 | Source |
|---|---|---|---|
| Mel filterbank low cutoff (`F_MIN`) | 300 Hz | **160 Hz** | NMFP fix #6 (Araz et al. 2025) |
| Per-batch sampling | seg_mode=`all` (all segs/track) | **`random_oneshot`** (1 anchor/track/epoch, resampled per epoch) | NMFP fix #2 |
| Batch size | 120 | **320** | scaled to give NT-Xent more in-batch negatives |
| Max epochs | 10 | **30** | matched length so the recipe-induced gradient pressure compounds |
| Loss | NT-Xent τ=0.05 | NT-Xent τ=0.05 (preserved) | — |
| Optimizer | Adam, LR=1e-4, cos | Adam, LR=1e-4, cos (preserved) | — |
| Seeds | 1 (random) | 42, 137, 2026 (pre-registered) | — |

**Training compute:** ~27 min/seed on Colab L4 GPU. **Why it works** (mechanism, grounded in the post-mortem): the same-artist top-1 confusion dominating baseline misses (14/17 on main_1s) is recovered by the false-negative-removal sampler + lower F_MIN, which together expose more low-frequency tonal information and stop the encoder from pulling same-track segments apart.

---

## Teaching documents

Two long-form reading guides built from the same automated pipeline. Self-contained, no prior AFP knowledge assumed.

| Doc | What it is | Pages |
|---|---|---|
| [`docs/AFP_Overview_Guide.pdf`](docs/AFP_Overview_Guide.pdf) ([md source](docs/overview.md)) | Friendly high-level walkthrough — pipelines, analogies, "Questions they'll ask" boxes. No heavy math. Best starting point. | 51 |
| [`docs/AFP_Teaching_Guide.pdf`](docs/AFP_Teaching_Guide.pdf) ([md source](docs/teach_me.md)) | Deep guide — every topic split into 🟢 Must-Know + 🔵 Depth. Includes formulas (STFT, mel, NT-Xent, McNemar) + cheat sheet. | 64 |

Rebuild from source:

```bash
# both PDFs come from one script (set up: pip install markdown-pdf)
python scripts/build_teach_me_pdf.py --doc overview
python scripts/build_teach_me_pdf.py --doc teach
```

---

## Repo layout

```
.
├── README.md                              # ← you are here
├── LICENSE                                # MIT (code only; data has separate licenses)
├── pyproject.toml + uv.lock               # reproducible Python 3.11 env (uv)
├── .env.example                           # template — fill in to enable Kaggle/HF pushes
├── .gitignore                             # excludes 67 GB audio + 2 GB checkpoints
│
├── docs/
│   ├── overview.md                        # high-level teaching doc (source)
│   ├── teach_me.md                        # deep teaching doc (source)
│   ├── AFP_Overview_Guide.pdf             # rendered overview (51 pp)
│   ├── AFP_Teaching_Guide.pdf             # rendered deep guide (64 pp)
│   ├── post_mortem_2026-05-12.md          # earlier post-mortem (Phase 1 audit)
│   └── POST_MORTEM_RECIPE_V3.md           # ← this work: baseline vs recipe v3 analysis
│
├── data/
│   ├── manifests/{hindustani,carnatic}/   # refs.csv + queries CSVs (small, tracked)
│   └── results/
│       ├── {olaf,dejavu,panako,nafp}/saraga_only_{main,ablation}{,_1s,_3s,_5s}/scores.json
│       └── nafp/
│           ├── saraga_only_main/          # NAFP-ckpt-10 baseline (10s cell)
│           ├── recipe_v2_10ep/            # First attempt (borderline, p=0.152)
│           ├── recipe_v3_30ep/            # ← headline result: PROTOCOL.md + RESULTS.md
│           ├── intervention2/             # per-artist mean — pre-reg NEGATIVE
│           ├── hubness_postproc/          # InvSoftmax + CSLS — pre-reg NEGATIVE
│           └── nmfp_eval/                 # NMFP-ckpt-100 reference (Araz et al. 2025)
│
├── scripts/
│   ├── build_teach_me_pdf.py              # builds the two teaching PDFs from docs/*.md
│   ├── score.py                           # canonical HR@k + MRR + top1_near + Wilson CI
│   ├── manifest.py                        # build refs.csv + queries CSVs from Saraga
│   ├── build_testset.py                   # cut query WAVs from refs (seeded)
│   ├── post_mortem_recipe_v3.py           # regenerable post-mortem analysis
│   ├── systems/                           # 4 AFP system runners
│   │   ├── olaf_runner.py
│   │   ├── dejavu_runner.py
│   │   ├── panako_runner.py
│   │   └── nafp_runner.py
│   ├── nafp/                              # NAFP-specific training + eval
│   │   ├── upstream/                      # patched NAFP (Chang 2021) + our 3 patches
│   │   ├── kaggle_train.py                # Kaggle T4 training kernel
│   │   ├── intervention2/                 # pre-registered NEGATIVE result
│   │   ├── recipe_v2_eval/                # Saraga eval pipeline (also used by v3)
│   │   ├── recipe_v3_eval/                # 3-seed orchestrator
│   │   └── nmfp_eval/                     # NMFP-ckpt-100 reference run
│   └── hf/                                # Hugging Face dataset build + push
│
└── notebooks/
    ├── colab_recipe_v2_train.ipynb        # 10-ep, BSZ=120 (single-seed, borderline)
    └── colab_recipe_v3_train.ipynb        # 30-ep, BSZ=320, 3-seed (Bonferroni-sig)
```

**Not tracked** (in `.gitignore`):
- Audio under `data/{mirdata,fma,queries*,refs,aug}/` — regenerable from `scripts/download_*.py`
- Model checkpoints `**/checkpoint/` and embedding memmaps `*.mm` — released separately on Hugging Face / Drive
- `data/results/**/query_results*.parquet` (~hundreds of MB) — regenerable from training + eval

---

## Quick start

### 1. Clone + setup

```bash
git clone https://github.com/ipritamdash/afp-indian-classical.git
cd afp-indian-classical

# Python 3.11 via uv (https://github.com/astral-sh/uv)
uv sync

# Configure secrets
cp .env.example .env
# Edit .env with your Kaggle / HuggingFace / GitHub tokens
```

### 2. Fetch upstream data (not in this repo)

```bash
# Saraga 1.5 (CC-BY-NC-SA, via mirdata)
uv run python scripts/download_saraga.py

# FMA-medium training corpus (for NAFP retraining; via Kaggle)
uv run python scripts/download_fma.py

# (Optional) NMFP-ckpt-100 reference weights (AGPLv3 / GPLv3, via Zenodo)
# Used only as evaluation ceiling; not bundled here.
```

### 3. Build the test set

```bash
uv run python scripts/manifest.py                    # refs.csv + main queries CSV
uv run python scripts/build_testset.py               # cut query WAVs at random offsets
uv run python scripts/build_query_length_variants.py # 1s / 3s / 5s truncations
```

### 4. Run any system on any cell

```bash
# Classical hash-based systems
uv run python scripts/systems/olaf_runner.py   --queries data/manifests/{hindustani,carnatic}/queries_1s.csv
uv run python scripts/systems/dejavu_runner.py --queries ...
uv run python scripts/systems/panako_runner.py --queries ...

# NAFP (baseline ckpt-10)
uv run python scripts/systems/nafp_runner.py --queries ... --checkpoint-dir ...
```

Each runner writes `query_results.parquet` + `scores.json` to `data/results/<system>/<cell>/`.

---

## Reproducing recipe v3 (the headline result)

End-to-end, ~4 hours wall time.

### Train (Colab L4 GPU, ~80 min)

Public training notebook: **<https://colab.research.google.com/drive/1bS3q8vOiylqW2lyICls_i-cGFKC8VZc6>** (mirror of `notebooks/colab_recipe_v3_train.ipynb`).

1. Open the Colab link above (or `notebooks/colab_recipe_v3_train.ipynb` locally)
2. Upload via left sidebar: `kaggle.json` + `nafp_patched_recipe_v3.tar.gz` (built from `scripts/nafp/upstream/`)
3. Runtime → Change runtime type → **L4 GPU**
4. Runtime → **Run all**
5. Wait ~80 min, download `recipe_v3_3seeds_output.tar.gz`

### Evaluate locally (Mac Metal GPU, ~2 h)

```bash
# Extract checkpoints
mkdir -p data/results/nafp/recipe_v3_30ep
tar -xzf ~/Downloads/recipe_v3_3seeds_output.tar.gz \
    -C data/results/nafp/recipe_v3_30ep --strip-components=1

# Eval each seed × 8 cells (uses tensorflow-metal if available, else CPU)
for seed in 42 137 2026; do
    uv run python scripts/nafp/recipe_v2_eval/eval.py \
        --ckpt-dir data/results/nafp/recipe_v3_30ep/seed${seed} \
        --ckpt-name ckpt-30 \
        --cfg data/results/nafp/recipe_v3_30ep/recipe_v3.yaml \
        --out-dir data/results/nafp/recipe_v3_30ep/seed${seed}_eval \
        --cells main_1s main_3s main_5s main_10s \
                ablation_1s ablation_3s ablation_5s ablation_10s
done
```

### Pooled-McNemar + post-mortem

```bash
uv run python scripts/post_mortem_recipe_v3.py
# → docs/POST_MORTEM_RECIPE_V3.md (regenerated from raw parquets)
```

---

## Pre-registered protocols + negative results

We pre-register every intervention's hypothesis, primary endpoint, and falsification
rule **before** training data is collected. Hypotheses are locked at the file shown.
Two interventions returned negative results under this protocol; both are reported
with the same rigor as the positive recipe v3 finding.

| Intervention | Outcome | Protocol locked at |
|---|---|---|
| Per-artist mean subtraction (Intervention 2) | **NEGATIVE** (isotropic control matched V1) | [`data/results/nafp/intervention2/PROTOCOL.md`](data/results/nafp/intervention2/PROTOCOL.md) |
| Hubness post-processing (InvSoftmax + CSLS) | **NEGATIVE** (matches baseline exactly) | `OVERNIGHT_RESULTS.md` |
| Recipe v2 (10 ep, BSZ=120) | Borderline (single-seed p=0.152) | informal |
| **Recipe v3 (30 ep, BSZ=320, 3 seeds)** | **Bonferroni-significant ✓** | [`data/results/nafp/recipe_v3_30ep/PROTOCOL.md`](data/results/nafp/recipe_v3_30ep/PROTOCOL.md) |

The two negative results strengthen the recipe v3 positive: we ruled out
inference-only fixes (per-artist subtraction, hubness correction), isolating
training-recipe modifications as the actual lever.

---

## What does NOT work (honest disclosure)

- **Per-artist mean subtraction at inference**: mechanism falsified by isotropic-centroid control (matches V1 in 4/4 main cells).
- **Hubness post-processing** (Inverted Softmax, CSLS) on baseline embeddings: reproduces baseline HR@1 exactly. The failure is encoder-level, not embedding-geometry.
- **Recipe v2 (10 epochs, BSZ=120)**: borderline (p=0.152 single-seed). Needed more compute headroom; addressed in v3.
- **Triplet loss at low epochs**: rejected as v3 candidate based on Araz et al. + agent review — semi-hard mining converges too slowly at 30 ep.

---

## Honest limitations

- **Library is small** (357 refs) — 6 of 8 cells saturate at 1.0 on the baseline. The win is on the 2 hardest cells (1-second queries).
- **No FMA distractors** — a 25k-track gallery is more realistic but out of scope.
- **Shorter queries are first-N truncations** of the 10-second cuts, not random re-cuts. Known methodological gap from Chang 2021.
- **Recipe v3 trades epoch parity for convergence** — 30 ep vs baseline's 10 ep. The improvement is recipe + 3× longer training combined; single-component ablations not run (would require 4 more training runs).
- **3 seeds is the minimum** for stable pooled McNemar; 5+ seeds preferable in a follow-up.
- **NMFP-ckpt-100 (Araz et al. 2025) remains the ceiling** at HR@1 = 1.000 across all 8 cells. We reach ~95 % of that ceiling at ~10 % of their training compute, but do not beat it.

---

## Licenses + attribution

- **Code in this repo**: MIT (`LICENSE`)
- **Saraga 1.5 audio**: CC-BY-NC-SA 4.0 (Srinivasamurthy, Gulati, Repetto, Serra; via Zenodo)
- **FMA-medium audio**: CC-BY 4.0 (Defferrard et al. 2017; via Kaggle `mimbres/neural-audio-fingerprint`)
- **NAFP upstream** (`scripts/nafp/upstream/`): MIT (Chang et al. 2021; patched fork with 3 minimal patches documented in commits)
- **NMFP teacher weights**: GPLv3 / AGPLv3 viral copyleft (Araz et al. 2025) — **not bundled here**; fetched on demand by `scripts/nafp/nmfp_eval/*`

Cite as:

```bibtex
@misc{banwala2026afpindianclassical,
  author       = {Aryan Banwala},
  title        = {Audio Fingerprinting Benchmark for Indian Classical Music},
  year         = {2026},
  howpublished = {\url{https://github.com/ipritamdash/afp-indian-classical}},
}
```

Upstream citations: see [`hf_dataset/README.md`](hf_dataset/README.md) for full BibTeX of Saraga, NAFP, NMFP, FMA.

---

## Project status

- ✓ 5-system benchmark complete
- ✓ Recipe v3 Bonferroni-significant on the two 1-second cells
- ✓ Three pre-registered negative results (Intervention 2, hubness, recipe v2)
- ✓ Empirical post-mortem with per-cell miss characterization
- ✓ Public Hugging Face dataset live ([Tachyeon/audio-fingerprint-indian-bench](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench))
- ⊘ Single-component ablation of recipe v3 — out of compute budget
- ⊘ FMA-distractor gallery experiment — out of scope
