---
title: AFP Indian Classical Music
emoji: 🎵
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.14.0
python_version: "3.11"
app_file: app.py
pinned: false
license: other
short_description: AFP demo on Saraga 1.5 Indian classical music
---

# Audio Fingerprinting for Indian Classical Music

Live demo of the **Recipe v3** NAFP-style audio fingerprint model on the
[Saraga 1.5](https://zenodo.org/records/4301737) Indian classical music corpus.

**What it does.** Upload (or record) any audio clip 1–30 seconds long. The
system retrieves the closest matching segment from 357 Saraga ref recordings
and tells you the artist, raaga, taal, and offset in seconds. If the query
isn't in the library, you get a calibrated "**No match in library**" verdict
instead of a wrong answer.

## Headline numbers

- HR@1 on Saraga main_1s: **0.995** (3-seed mean) vs baseline NAFP-ckpt-10 at 0.983
- ~68 % miss-rate reduction across the 8-cell benchmark
- Pre-registered Bonferroni-significant (pooled McNemar **p = 3.18 × 10⁻⁶**)

## Out-of-library threshold (calibrated)

- **T = 0.7867** chosen via pre-registered protocol: smallest T with FPR ≤ 5 %
  on 167 random FMA-medium probes
- At chosen T: TPR = **99.09 %** (in-library queries correctly accepted)
- High-confidence threshold **T = 0.8918** (FPR ≤ 1 %, TPR = 85.4 %)

## What's running here

- **Encoder**: NAFP CNN (Chang et al. 2021) with 2 NMFP recipe fixes (Araz et al. 2025),
  trained from scratch on FMA-medium for 30 epochs at BSZ 320, seed 42
- **Library**: 690,414 Saraga ref segments at 0.5 s hop
- **Index**: FAISS IndexFlatIP (exact inner-product = cosine on unit-norm vectors)
- **Calibration**: 200 random FMA probes; ROC-based threshold; pre-registered

## Honest limitations

- **357 refs only.** A real Shazam indexes millions; this is a research-scale demo.
- **Phone-mic noise robustness not benchmarked.** Use clean audio for best results.
- **Mobile mic recording on iOS Safari is unreliable.** Use file upload.
- **Same-artist confusion still present** for the few queries that fail.

## License

- **Code (`app.py`)**: MIT
- **Model weights (`artifacts/ckpt-30.*`)**: MIT (trained on FMA-medium)
- **Reference embeddings (`artifacts/ref_embs.mm`)**: CC-BY-NC-SA 4.0
  (derivative of Saraga 1.5 audio). **Non-commercial use only.**

## Related

- 📦 Public dataset: [Tachyeon/audio-fingerprint-indian-bench](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench)
- 💻 Source code: [github.com/ipritamdash/afp-indian-classical](https://github.com/ipritamdash/afp-indian-classical) (private)

## Citations

```bibtex
@dataset{srinivasamurthy2021saraga,
  author = {Srinivasamurthy, Ajay and Gulati, Sankalp and Repetto, Rafael Caro and Serra, Xavier},
  title = {{Saraga}: Open Datasets for Research on {I}ndian Art Music},
  year = {2021}, version = {1.5}, publisher = {Zenodo},
  doi = {10.5281/zenodo.4301737}
}
@inproceedings{chang2021nafp,
  author = {Chang, Sungkyun and Lee, Donmoon and Park, Jeongsoo and Lim, Hyungui and Lee, Kyogu and Ko, Karam and Han, Yoonchang},
  title = {{Neural Audio Fingerprint} for High-specific Audio Retrieval based on Contrastive Learning},
  booktitle = {ICASSP}, year = {2021}, doi = {10.1109/ICASSP39728.2021.9414083}
}
@inproceedings{araz2025nmfp,
  author = {Araz, R. O. and Cortès-Sebastià, G. and Molina, E. and Serra, X. and Serra, J. and Mitsufuji, Y. and Bogdanov, D.},
  title = {Enhancing Neural Audio Fingerprint Robustness to Real-World Conditions},
  booktitle = {ISMIR}, year = {2025}, eprint = {arXiv:2506.22661}
}
```
