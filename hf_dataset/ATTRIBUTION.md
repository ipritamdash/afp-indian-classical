# Attribution

This dataset is a **derivative work** of the Saraga 1.5 corpus of Indian Art Music.
All audio query clips were cut from Saraga recordings; reference-track metadata,
section annotations, work keys, and the composition-twin leakage analysis are derived
directly from Saraga's mirdata-formatted annotations. We acknowledge the original
authors, collectors, performers, and the broader research programme that produced the
source corpus.

## Source corpus

> Bozkurt, B.; Srinivasamurthy, A.; Gulati, S.; Serra, X. (2018).
> *Saraga: research datasets of Indian Art Music* (v1.5).
> Zenodo. https://doi.org/10.5281/zenodo.4301737

Licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**
(CC-BY-NC-SA 4.0). This release preserves both the Non-Commercial and Share-Alike
obligations on every audio and metadata artefact derived from it.

## Research programme

The Saraga corpus was produced under the **CompMusic** project at the
**Music Technology Group**, Universitat Pompeu Fabra, Barcelona — an EU-ERC funded
research programme (2011–2016) on the computational analysis of non-Western art
musics, led by Prof. Xavier Serra.

- MTG: https://www.upf.edu/web/mtg
- CompMusic: https://compmusic.upf.edu/

## Performers

The Saraga 1.5 corpus contains studio and concert recordings by numerous Indian classical
musicians who consented to their performances being shared for non-commercial research
use. We acknowledge them as the originating artists; specific per-track attribution
(performing artist + concert) is preserved in this release at
`data/refs.parquet` (column `artists`) and `data/inspection/tracks.parquet` (columns
`artists`, `concert`, `track_mbid`). Researchers using this benchmark should
acknowledge artists in any downstream presentation or publication that names individual
tracks.

## What this release ADDS to the source

This dataset adds, on top of the Saraga corpus:

1. **A reproducible test-set design** — sample-accurate 10 s query cuts with seeded
   offsets (`GLOBAL_SEED=20260511`) and ±0.5 s jitter, sufficient for any audio
   fingerprinting evaluation that needs ground-truth offset precision.
2. **A composition-twin leakage analysis** — pairs of recordings sharing a MusicBrainz
   work-MBID flagged so that "the system retrieved a *different* performance of the
   same piece" can be separated from genuine identity matches in evaluation.
3. **Section-aligned ablation queries** — bucketed via diacritic-normalised keyword
   matching into `alaap` (improvisational), `composed` (rhythm body), and `tani`
   (percussion-only) — enables fine-grained per-section difficulty reporting.
4. **End-to-end baselines** for three open-source audio fingerprinters
   (Olaf, Dejavu, Panako) — raw retrieval rankings and scored metrics for both main
   and ablation splits.

These additions are themselves licensed CC-BY-NC-SA 4.0, matching the source.

## Citing this benchmark

If you use this dataset in academic work, please cite **both** the Saraga source corpus
(BibTeX above) **and** this derivative release; see `README.md` for the recommended
`@dataset` BibTeX entry for this release.
