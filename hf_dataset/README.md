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
- config_name: results_olaf
  data_files:
  - split: main
    path: data/results/olaf_main.parquet
  - split: ablation
    path: data/results/olaf_ablation.parquet
- config_name: results_dejavu
  data_files:
  - split: main
    path: data/results/dejavu_main.parquet
  - split: ablation
    path: data/results/dejavu_ablation.parquet
- config_name: results_panako
  data_files:
  - split: main
    path: data/results/panako_main.parquet
  - split: ablation
    path: data/results/panako_ablation.parquet
- config_name: results_nafp
  data_files:
  - split: main
    path: data/results/nafp_main.parquet
  - split: ablation
    path: data/results/nafp_ablation.parquet
- config_name: inspection_tracks
  data_files:
  - split: library
    path: data/inspection/tracks.parquet
- config_name: inspection_sections
  data_files:
  - split: library
    path: data/inspection/sections.parquet
- config_name: inspection_works
  data_files:
  - split: library
    path: data/inspection/works.parquet
- config_name: inspection_leakage_pairs
  data_files:
  - split: library
    path: data/inspection/leakage_pairs.parquet
---

# Audio Fingerprinting Benchmark on Indian Classical Music

A reproducible benchmark for **audio-fingerprint-based content identification** on
**Indian classical music** (Hindustani + Carnatic), built on top of the
[Saraga 1.5 corpus](https://zenodo.org/records/4301737) (MTG, Universitat Pompeu Fabra).

This release contains the **derivative artefacts** of the benchmark — 10-second query
clips with sample-accurate ground-truth offsets, reference-track metadata (no source
audio), per-system result parquets, and inspection tables (tracks / sections / works /
composition-twin leakage pairs). The **source MP3s are NOT redistributed**; rebuild the
library by fetching Saraga 1.5 directly from Zenodo.

## TL;DR

- **1 632 query clips** (1 000 main, 632 section-aligned ablation), 10 s mono 16 kHz,
  drawn from 357 Saraga concert recordings (108 Hindustani + 249 Carnatic).
- **Sample-accurate ground truth**: query offset, seed, source ref_id — alignment error
  measurable to sub-50 ms.
- **Composition-twin leakage analysis**: 165 / 1 000 main queries share a *MusicBrainz
  work-MBID* with a different recording in the reference library — flagged as a signal
  (kept and reported separately), not silently dropped.
- **Four fingerprinters baselined** end-to-end at 1 s / 3 s / 5 s / 10 s query lengths:
  Olaf, Dejavu, Panako (CQT triplet-hash), and **NAFP (Chang et al. ICASSP 2021)** trained
  on Mimbres' 10 k Western-pop dataset and zero-shot transferred to Indian classical.

## What's in this dataset

```
data/
├── queries/                          # AudioFolder pattern
│   ├── metadata.parquet              # 1 000 rows
│   ├── hindustani/*.wav              # 500
│   └── carnatic/*.wav                # 500
├── queries_ablation/
│   ├── metadata.parquet              # 632 rows; section_type ∈ {alaap, composed, tani}
│   ├── hindustani/*.wav              # 207
│   └── carnatic/*.wav                # 425
├── refs.parquet                      # 357 ref tracks; metadata only (NO audio paths)
├── results/                          # baselines (3 classical systems × main+ablation)
│   ├── {system}_{main,ablation}.parquet           # ranked candidates per query
│   └── {system}_{main,ablation}.scores.json       # HR@k + Wilson 95% CI + alignment error
├── inspection/
│   ├── tracks.parquet                # combined H+C track index (357 rows)
│   ├── sections.parquet              # 749 section annotations
│   ├── works.parquet                 # 616 work keys (mbid + title-lower)
│   └── leakage_pairs.parquet         # 123 composition-twin pairs
└── configs/                          # seeded test-set generation configs (JSON, for repro)
```

`ARTEFACTS.parquet` at the root lists every staged file with its `sha256` — verify
the upload integrity after download.

## How to load

```python
from datasets import load_dataset

# 10 s query clips with ground-truth offset + source ref_id
queries = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "queries", split="test")
queries[0]
# {'audio': {...}, 'query_id': 'carnatic_t0000_q0', 'ref_id': 'carnatic_0_…',
#  'offset_sec': 514.2, 'length_sec': 10.0, 'seed': 993916075,
#  'has_twin_in_library': False, 'target_sr': 16000, 'target_channels': 1}

# Section-aligned ablation (alaap / composed / tani)
abl = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "queries_ablation", split="test")

# Reference-track metadata (no audio — pull MP3s from Zenodo)
refs = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "refs", split="library")

# Per-system raw retrieval results — one row per (query, rank)
olaf_results = load_dataset("Tachyeon/audio-fingerprint-indian-bench", "results_olaf",
                            split="main")
```

## Methodology

### Test-set generation

Per corpus, the longest 100 concert recordings (or all if fewer) are designated
*test-source* tracks. From each track, **5 query offsets** are sampled uniformly between
`skip_head_sec=10` and `duration − skip_tail_sec − longest_query − jitter`, perturbed by
±0.5 s seeded jitter (`GLOBAL_SEED=20260511`). Each offset yields a single 10-second cut
at 16 kHz / mono / PCM_16 via `soundfile` for sample-accurate alignment.

Total: 500 main queries per corpus × 2 corpora = **1 000 main queries**.

### Ablation (section-aligned)

For tracks with Saraga section annotations, queries are drawn *inside* a single section
(no boundary crossing). Section labels are NFKD-normalised + diacritic-stripped before
needle matching, so `Ṭhumri`, `Caraṇaṁ`, `Khyāl`, etc. classify reliably. Buckets:

- **`alaap`**: improvisational pulseless opening (`ālāp`, `alapana`, …)
- **`composed`**: composition body with lyrics + rhythm (`khyāl`, `pallavi`, `kriti`,
  `caraṇam`, `tarānā`, `bandish`, `thumri`, `bhajan`, `dadra`, …)
- **`tani`**: percussion-only avartana (Carnatic only; difficulty control)

Distribution:

| corpus      | alaap | composed | tani | total |
|-------------|-------|----------|------|-------|
| hindustani  | 67    | 140      | 0    | 207   |
| carnatic    | 70    | 344      | 11   | 425   |
| **total**   | 137   | 484      | 11   | 632   |

Sections labelled `Laggī` (Hindustani tabla percussion finale) and intermediate-onset
forms (Kalpanā svara, Neraval, Tānam) are intentionally not bucketed — they sit
between alaap and composed and would muddy the section-effect contrast.

### Composition-twin leakage

Two recordings of the **same composition** (matched on MusicBrainz `work-mbid`, or
fuzzy lowercased title fallback) yield highly self-similar acoustic content. For audio
fingerprinting this is *not* identity — the systems are not expected to match a
different performance — but composition similarity can leak into match scores.

`inspection/leakage_pairs.parquet` lists all 123 twin pairs; `queries.parquet` carries a
boolean `has_twin_in_library` per query (165 / 1 000 main queries flagged).
`scores.json` per system reports HR@k split by `with_twin` vs `no_twin` so the leakage
contribution can be audited rather than hidden.

**Caveat to note before quoting twin rates**: 47 / 108 Hindustani refs and 26 / 249
Carnatic refs have no `works`/`work_mbids` metadata, so twin detection silently treats
them as "no twin" — the reported leakage rate is a **lower bound only**. Additionally,
the Hindustani library contains only 8 `library_only` refs (the rest are also
`test_source`), so for Hindustani the flag effectively measures
test-source ↔ test-source collisions, not retrieval-confounding library matches.

## Baseline results (Saraga library, no FMA distractors)

Top-1 hit rate with Wilson 95 % CI; alignment error vs ground truth.

**10-second queries** (main = 1 000 queries; ablation per-section = 632 queries):

| system  | HR@1 main (n=1 000) | MRR@10 | top1_near | HR@1 alaap (n=137) | HR@1 composed (n=484) | HR@1 tani (n=11) | median align err |
|---------|---------------------|--------|-----------|--------------------|------------------------|------------------|------------------|
| Olaf    | 0.998 [.993, 1.00]  | 0.998  | 0.998     | 1.000              | 1.000                  | 1.000            | 249 ms           |
| Dejavu  | 1.000 [.996, 1.00]  | 1.000  | 1.000     | 1.000              | 1.000                  | 1.000            | 12 ms            |
| Panako† | 0.997 [.991, .999]  | 0.997  | 0.997     | 0.985              | 0.994                  | 1.000            | 3 ms             |
| **NAFP**| **1.000 [.996, 1.00]** | **1.000** | **1.000** | **1.000**     | **1.000**              | **1.000**        | 124 ms           |

† Panako retuned with `PANAKO_MIN_MATCH_DURATION=0.5, MIN_HITS_FILTERED=2, MIN_HITS_UNFILTERED=3`
to expose its sub-5-s capability; with the upstream default (`MIN_MATCH_DURATION=5`)
Panako returns 0 on all queries < 5 s.

**Length-degradation** (main queries; HR@1 with Wilson 95 % CI):

| system  | 1 s                 | 3 s                 | 5 s                 | 10 s                |
|---------|---------------------|---------------------|---------------------|---------------------|
| Olaf    | 0.492 [.461,.523]   | 0.955 [.940,.966]   | 0.993 [.986,.997]   | 0.998 [.993,1.00]   |
| Dejavu  | 0.745 [.717,.771]   | 0.969 [.956,.978]   | 0.994 [.987,.997]   | 1.000 [.996,1.00]   |
| Panako† | 0.000 [.000,.004]   | 0.000 [.000,.004]   | 0.922 [.904,.937]   | 0.997 [.991,.999]   |
| **NAFP**| **0.983 [.973,.989]** | **0.998 [.993,1.00]** | **0.999 [.994,1.00]** | **1.000 [.996,1.00]** |

Notes:
- **NAFP wins at every length on main queries.** At 1 s it leads the next-best system
  (Dejavu) by 23.8 percentage points (0.983 vs 0.745). The lead narrows to 0.5 pp at 5 s
  and ties at 1.000 at 10 s with Dejavu.
- **Panako has a hard structural floor at ~5 s** — even with retuned thresholds it
  returns 0 on queries < 5 s. Below the CQT-fingerprinting design point.
- **Dejavu beats Olaf at short query lengths** among hash systems.
- **Alaap is not harder than composed at short lengths** for hash systems — actually
  marginally easier (e.g. Dejavu 1 s ablation: alaap 0.825 vs composed 0.746).
- Olaf's median alignment error is offset-quantised to its event-hop, not a defect.
- Tani N = 11 is small; treat its per-cell HR as a probe, not a reportable estimate
  (95 % Wilson CI on HR = 1.0 is [0.74, 1.00]).
- NAFP zero-shot transfer from Western-pop training (Mimbres 10 k) generalises cleanly
  to Indian classical at recording-identity retrieval — no Indian-classical training data
  was used.

## Reproducibility

- `data/configs/{corpus}_{main,ablation}.json` carries `global_seed=20260511`, query length,
  per-corpus track counts, and the alaap/composed/tani keyword lists used for section
  bucketing.
- All seeds are deterministic given Python's `random.Random(seed)` + the global seed.
- Source-audio MD5s are recorded in `refs.parquet` (`source_md5`) so a downstream
  consumer can verify they pulled the same Saraga 1.5 release from Zenodo.

## Limitations

- **Library is Saraga-only**: no Western distractors are mixed into the reference set
  in this release. A separate FMA-medium distractor experiment is planned.
- **Tani only present in Carnatic** (11 ablation queries) — too few for stand-alone
  conclusions; included as a difficulty *probe*, not a reportable cell.
- **Per-artist concentration in ablation**: Sanjay Subrahmanyan owns 143 / 425 Carnatic
  ablation queries (33.7 %); top-3 Carnatic artists own 53.5 %. Hindustani similar
  (Ajoy Chakrabarty 24 % of test-source tracks). Per-section HR differences may partly
  reflect singer-style differences; users should report per-artist HR alongside per-section
  HR when making section-effect claims.
- **Composition twins are a feature, not a bug**: HR@k may exceed the "true" identity
  rate when systems retrieve a twin recording. `with_twin` / `no_twin` splits in
  `scores.json` separate this.
- **Query offsets cover the central 66 %** of every track (skip-head 10 s + skip-tail 10 s
  + query length 10 s + jitter). Queries are *not* uniformly distributed over [0, 1] of
  track duration.
- Source MP3s are not redistributed — fetch from Zenodo to rebuild the library.

## Licensing

| component                | license                                          |
|--------------------------|--------------------------------------------------|
| Query audio (cut WAVs)   | **CC-BY-NC-SA 4.0** (inherited from Saraga 1.5)  |
| Metadata, results        | **CC-BY-NC-SA 4.0** (Share-Alike preservation)   |
| Generating scripts*      | MIT — see `LICENSE-CODE`                         |

\* The benchmark generation / scoring scripts are an independent work; we license the
**code** permissively (MIT) but the **data products** (cuts, manifests, results) keep
Saraga's share-alike obligation. Commercial use of the audio is **not** permitted.

## Attribution

The source corpus is **Saraga 1.5**:

> Bozkurt, B.; Srinivasamurthy, A.; Gulati, S.; Serra, X. (2018).
> *Saraga: research datasets of Indian Art Music* (v1.5).
> Zenodo. https://doi.org/10.5281/zenodo.4301737

See `ATTRIBUTION.md` for the full chain (MTG, CompMusic, artists) and per-artist credits.

## Citation

If you use this benchmark, please cite both the underlying corpus **and** this release:

```bibtex
@dataset{bozkurt2018saraga,
  author    = {Bozkurt, Bar{\i}{\c{s}} and Srinivasamurthy, Ajay and
               Gulati, Sankalp and Serra, Xavier},
  title     = {Saraga: research datasets of Indian Art Music},
  year      = {2018},
  version   = {1.5},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4301737}
}

@dataset{tachyeon2026afpindbench,
  author    = {Dash, Pritam},
  title     = {Audio Fingerprinting Benchmark on Indian Classical Music},
  year      = {2026},
  version   = {v0.5},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench}
}
```

## Changelog

- **v0.5** (2026-05-12): post-mortem audit fixes. `has_twin_in_library` recomputed
  from `work_mbids` (was previously derived from works-text, which over-flagged
  generic form-names like "Thillana" and "Ragam Thanam Pallavi"). Twin count
  165 → 130 (35 spurious flags eliminated). Per-system twin/no-twin numbers
  updated; headline HR@1 unchanged. Dejavu parquets' `ref_stop` and `query_stop`
  columns corrected to use actual query length (1/3/5/10 s) instead of a hardcoded
  10.0 s — fixes silent length-mismatch on the short-query splits. `score.py`
  `no_match` denominator switched to manifest-derived (correctly counts queries
  absent from results.parquet as no-match). `nafp_runner.py` uses
  `np.argsort(kind='stable')` for deterministic tie-breaking. All 32 cells
  re-scored; sanity invariants hold (top1_near ≤ HR@1; MRR@10 ≥ HR@1 in 32/32 cells).
  See `docs/post_mortem_2026-05-12.md` in the project repo for full audit detail.
- **v0.4** (2026-05-12): **NAFP added as the 4th system.** NAFP trained on Mimbres'
  10 k Western-pop 30-s segments on Kaggle T4 (10 epochs, ckpt-10, NT-Xent τ=0.05,
  TR_BATCH_SZ=120, TR_N_ANCHOR=60); zero-shot transferred to Saraga 1.5. Mac inference
  via FAISS IndexFlatIP (cosine via IP on L2-normalised 128-D embeddings), sequence-level
  scoring `mean(diag(Q · R.T))` per Chang 2021 §3.4. Result parquets and `scores.json`
  added at `data/results/nafp_{main,ablation}.{parquet,scores.json}`. NEW METRICS added
  to every system's scores.json: `mrr@5`, `mrr@10`, `top1_near` (±0.5 s tolerance, rate
  over n_queries — matches NAFP paper's "top-1 near" definition), plus `top1_near_at_0.05s`
  and `top1_near_at_1.0s`. Panako results re-scored under retuned config
  (`PANAKO_MIN_MATCH_DURATION=0.5`); previous default-config Panako results are preserved
  at `data/results/panako_*.default-config.{parquet,scores.json}`.
- **v0.3** (2026-05-12): refreshed ablation result parquets + scores against the
  current 632-query ablation manifest (previous result files were against the v0.1
  624-query set). Baseline results table now reflects the v0.2 manifest. Added
  length-degradation table at 1 s / 3 s / 5 s for all three systems against the main
  query set. Main result parquets unchanged (1 000-query main set unchanged across
  releases).
- **v0.2** (2026-05-12): expanded section-aligned ablation coverage — Hindustani
  composed bucket now includes Ṭhumri, Bhajan, Dādrā genres (NFKD-normalised section
  label matching); Carnatic composed includes the multi-diacritic Caraṇaṁ variant.
  Ablation count: 624 → **632** (Hindustani 200 → 207, Carnatic 424 → 425). 0 changes
  to main queries, 0 changes to existing classifications.
- **v0.1** (2026-05-12): initial release with 3 classical fingerprinters baselined;
  NAFP and FMA-distractor extension pending.
