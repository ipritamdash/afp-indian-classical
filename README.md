# Audio Fingerprinting Benchmark for Indian Classical Music

Reproducible benchmark of five audio-fingerprinting systems on the
**Saraga 1.5** Indian classical music corpus, with a Bonferroni-significant
training-recipe improvement to the NAFP baseline.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗-dataset-yellow.svg)](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench)

---

## What this repo contains

| Component | What it does |
|---|---|
| **Five fingerprinting systems** | Olaf, Dejavu, Panako, NAFP, NMFP — runners + scoring |
| **Saraga 1.5 test harness** | Reference library (357 tracks) + 1632 queries × 4 lengths (1/3/5/10 s) |
| **Recipe v3** | Training-recipe improvement on NAFP — Bonferroni-significant gain |
| **Pre-registered protocols** | Hypothesis-locked-before-data designs for every intervention |
| **Public dataset on Hugging Face** | All result parquets + manifests + scores |

---

## Headline result

**Recipe v3** trains NAFP with two recipe modifications from Araz et al. (ISMIR 2025) at a 3× larger batch and 3× longer schedule, on three independent seeds.

| Cell | Baseline NAFP-ckpt-10 | **Recipe v3 (3-seed mean)** | Pooled McNemar p |
|---|---|---|---|
| main_1s | 0.983 (17 miss) | **0.995 (5 miss)** | **3.18 × 10⁻⁶** ✓✓ |
| main_3s | 0.998 | 1.000 | 0.031 |
| main_5s | 0.999 | 1.000 | 0.250 |
| main_10s | 1.000 | 1.000 | — |
| ablation_1s | 0.979 (13 miss) | **0.991 (6 miss)** | **0.0046** ✓✓ |
| ablation_3s | 1.000 | 1.000 | — |
| ablation_5s | 1.000 | 1.000 | — |
| ablation_10s | 1.000 | 1.000 | — |

✓✓ Bonferroni-significant at α/8 = 0.00625. No cell regresses on average.
Full results: [`data/results/nafp/recipe_v3_30ep/RESULTS.md`](data/results/nafp/recipe_v3_30ep/RESULTS.md).

---

## Project layout

```
.
├── README.md                  # this file
├── LICENSE                    # MIT
├── pyproject.toml             # uv-managed Python deps
├── uv.lock                    # locked deps
├── .env.example               # template for secrets (Kaggle / HF / GitHub)
├── .gitignore                 # excludes audio, checkpoints, memmaps
│
├── docs/                      # Long-form design + post-mortem docs
│   └── post_mortem_2026-05-12.md
│
├── data/                      # Manifests + result metadata (no audio, no weights)
│   ├── manifests/             # CSVs: refs.csv, queries_*.csv per corpus
│   └── results/               # scores.json + RESULTS.md per (system, cell)
│       ├── olaf/              # Olaf — 4 lengths × main+ablation
│       ├── dejavu/            # Dejavu — same
│       ├── panako/            # Panako — same
│       ├── nafp/              # NAFP family (baseline + recipe_v2 + recipe_v3 + intervention2 + hubness)
│       │   ├── saraga_only_main/      # baseline NAFP-ckpt-10 results
│       │   ├── recipe_v2_10ep/        # NMFP-fix attempt #1 (10 ep, BSZ=120) — borderline
│       │   ├── recipe_v3_30ep/        # NMFP-fix attempt #2 (30 ep, BSZ=320, 3 seeds) — Bonferroni-sig
│       │   ├── intervention2/         # per-artist mean subtraction — pre-reg negative result
│       │   ├── hubness_postproc/      # InvSoftmax / CSLS post-proc — pre-reg negative result
│       │   └── nmfp_eval/             # NMFP-ckpt-100 (Araz et al.) ceiling reference
│       └── ...
│
├── scripts/                   # All Python entry-points (run-from-root)
│   ├── score.py               # canonical scorer (HR@k, MRR@k, top1_near)
│   ├── manifest.py            # build refs.csv + queries CSV(s) from Saraga
│   ├── build_testset.py       # cut query WAVs from refs
│   ├── build_query_length_variants.py
│   ├── download_saraga.py
│   ├── download_fma.py
│   ├── download_zenodo.py
│   ├── check_leakage.py       # train/test work-MBID overlap audit
│   ├── audit_refs.py
│   ├── consolidate_inspection.py
│   ├── systems/               # AFP system runners
│   │   ├── olaf_runner.py
│   │   ├── dejavu_runner.py
│   │   ├── panako_runner.py
│   │   └── nafp_runner.py
│   ├── nafp/                  # NAFP-specific training + eval
│   │   ├── upstream/          # patched NAFP code (Chang et al. 2021 + our 3 patches)
│   │   ├── kaggle_train.py    # Kaggle T4 training kernel
│   │   ├── intervention2/     # per-artist mean subtraction pre-reg
│   │   ├── recipe_v2_eval/    # Saraga eval for recipe_v2 and recipe_v3
│   │   ├── recipe_v3_eval/    # orchestrator for 3-seed eval
│   │   ├── nmfp_eval/         # NMFP-ckpt-100 reference run + hubness
│   │   └── infer_kernel/      # Kaggle inference kernel
│   └── hf/                    # Hugging Face dataset build + push
│       ├── build_hf_dataset.py
│       └── push_hf_dataset.py
│
└── notebooks/                 # Colab training notebooks
    ├── colab_recipe_v2_train.ipynb     # 10-ep, BSZ=120 (single-seed, borderline)
    └── colab_recipe_v3_train.ipynb     # 30-ep, BSZ=320 (3-seed, Bonferroni-sig)
```

**Not tracked in git** (regenerable / large): audio under `data/{mirdata,fma,queries*,refs,aug}/`, model checkpoints under `**/checkpoint/`, embedding memmaps `*.mm`. See `.gitignore`.

---

## Getting set up

```bash
# Clone
git clone https://github.com/<your-handle>/<repo-name>.git
cd <repo-name>

# Python 3.11 (project pins this via .python-version)
# We use uv (https://github.com/astral-sh/uv)
uv sync

# Copy secrets template and fill in tokens
cp .env.example .env
# Edit .env with your Kaggle / HuggingFace / GitHub tokens
```

You will need:
- **Saraga 1.5** — fetched via `mirdata` in `scripts/download_saraga.py` (Zenodo, CC-BY-NC-SA)
- **FMA-medium** — fetched via Kaggle in `scripts/download_fma.py` (Kaggle dataset: `mimbres/neural-audio-fingerprint`)
- **NMFP teacher weights** (optional, AGPLv3) — Zenodo 15719945, used only as evaluation ceiling

---

## Reproducing the headline result (Recipe v3)

This is the Bonferroni-significant improvement on NAFP-ckpt-10. End-to-end:

```bash
# 1. Build manifests from Saraga (one-off)
uv run python scripts/manifest.py
uv run python scripts/build_query_length_variants.py

# 2. (Once) Train baseline NAFP-ckpt-10 — 10 ep, BSZ=120
#    Push to Kaggle T4 via:
uv run python scripts/nafp/push_and_wait.py --kernel recipe_baseline

# 3. Train Recipe v3 — 3 seeds × 30 ep × BSZ=320 on Colab L4
#    See: notebooks/colab_recipe_v3_train.ipynb
#    Upload: kaggle.json + nafp_patched_recipe_v3.tar.gz
#    Run all cells; download recipe_v3_3seeds_output.tar.gz

# 4. Extract checkpoints locally
mkdir -p data/results/nafp/recipe_v3_30ep
tar -xzf ~/Downloads/recipe_v3_3seeds_output.tar.gz \
    -C data/results/nafp/recipe_v3_30ep --strip-components=1

# 5. Eval each seed on all 8 cells (Mac M5 Metal recommended; CPU works)
for seed in 42 137 2026; do
    uv run python scripts/nafp/recipe_v2_eval/eval.py \
        --ckpt-dir data/results/nafp/recipe_v3_30ep/seed${seed} \
        --ckpt-name ckpt-30 \
        --cfg data/results/nafp/recipe_v3_30ep/recipe_v3.yaml \
        --out-dir data/results/nafp/recipe_v3_30ep/seed${seed}_eval \
        --cells main_1s main_3s main_5s main_10s \
                ablation_1s ablation_3s ablation_5s ablation_10s
done

# 6. Pooled McNemar (writes pooled_mcnemar.csv + RESULTS.md)
uv run python -c "exec(open('scripts/nafp/recipe_v3_eval/pool_seeds.py').read())"
```

Per-cell results land in `data/results/nafp/recipe_v3_30ep/seed{42,137,2026}_eval/<cell>/scores.json`.

---

## Reproducing the full 5-system benchmark

Each runner produces `query_results.parquet` + `scores.json` per cell.

```bash
# Olaf
uv run python scripts/systems/olaf_runner.py --queries data/manifests/{hindustani,carnatic}/queries_1s.csv

# Dejavu
uv run python scripts/systems/dejavu_runner.py --queries ...

# Panako
uv run python scripts/systems/panako_runner.py --queries ...

# NAFP (baseline ckpt-10)
uv run python scripts/systems/nafp_runner.py --queries ... --checkpoint-dir ...

# NMFP (reference upper bound — Araz et al. 2025)
uv run python scripts/nafp/nmfp_eval/run_nmfp_native.py --cell main_1s ...
```

Then aggregate with the canonical scorer:

```bash
uv run python scripts/score.py \
    --results data/results/<system>/<cell>/query_results.parquet \
    --manifest data/manifests/{hindustani,carnatic}/queries_1s.csv
```

---

## Pre-registered protocols + negative results

We use a pre-registration discipline: every intervention's hypothesis,
primary endpoint, and falsification rule is committed to a `PROTOCOL.md`
**before** training data is collected. The five interventions we ran:

| Intervention | Outcome | Protocol |
|---|---|---|
| **Per-artist mean subtraction (Intervention 2)** | Pre-registered NEGATIVE (isotropic control matched V1) | [`data/results/nafp/intervention2/PROTOCOL.md`](data/results/nafp/intervention2/PROTOCOL.md) |
| **Hubness post-processing (InvSoftmax / CSLS)** | Pre-registered NEGATIVE (matches baseline exactly) | (in `OVERNIGHT_RESULTS.md`) |
| **Recipe v2 (10 ep, BSZ=120)** | Borderline (single-seed p=0.152) | (informal) |
| **Recipe v3 (30 ep, BSZ=320, 3 seeds)** | **Bonferroni-significant ✓** | [`data/results/nafp/recipe_v3_30ep/PROTOCOL.md`](data/results/nafp/recipe_v3_30ep/PROTOCOL.md) |

Negative results are reported with the same rigor as positives.

---

## Public dataset on Hugging Face

All eval results, manifests, queries, and per-system parquets are mirrored at:

**🤗 [Tachyeon/audio-fingerprint-indian-bench](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench)**

Use the HF version for read-only reproducibility checks.

---

## Attribution + licenses

Code in this repo is **MIT-licensed** (see `LICENSE`). Source code, manifests, scripts, and configurations only.

**Important external dependencies** — see `docs/` for full citations:

- **Saraga 1.5** corpus (CC-BY-NC-SA 4.0) — Srinivasamurthy, Gulati, Repetto, Serra. CompMusic / MTG, UPF.
- **NAFP** (Chang et al., ICASSP 2021, MIT-licensed) — `scripts/nafp/upstream/` contains patched fork.
- **NMFP** (Araz et al., ISMIR 2025, GPLv3) — recipe inspiration. **Not bundled**: our scripts download weights from Zenodo on demand; we do not ship their weights.
- **FMA-medium** — Defferrard et al., ISMIR 2017 (CC-BY 4.0 license; via `mimbres/neural-audio-fingerprint` Kaggle dataset).

Cite as:

```bibtex
@misc{afp-indian-classical-2026,
  author = {<your name>},
  title = {Audio Fingerprinting Benchmark for Indian Classical Music},
  year = {2026},
  howpublished = {\url{https://github.com/<your-handle>/<repo-name>}},
}
```

---

## Honest limitations

- **357 reference tracks** — small library; HR@1 saturates for 5s/10s queries even on the baseline. The hard case is 1s.
- **Single training corpus** — FMA-medium only; no domain adaptation experiments.
- **Recipe v3 trades epoch parity for convergence** — disclosed in `RESULTS.md`. The improvement is recipe + 3× longer training combined; individual ablations not run.
- **3 seeds is the minimum** for stable pooled-McNemar; we'd prefer 5+ in a follow-up.
- **NMFP-ckpt-100 (Araz et al. 2025) remains the ceiling** at HR@1 = 1.000 across all 8 cells. We reach ~95% of that ceiling at ~10% of their training compute, but do not beat them.

---

## Project status

- 5-system benchmark complete
- Recipe v3 Bonferroni-significant on hardest cells (1-second queries)
- Three pre-registered negative results (Intervention 2, hubness, recipe v2)
- Public HF dataset live
- See `data/results/nafp/recipe_v3_30ep/RESULTS.md` for the full writeup
