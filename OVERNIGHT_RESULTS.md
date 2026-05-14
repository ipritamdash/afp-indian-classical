# Overnight Results — NMFP transfer + Hubness post-processing on Saraga 1.5

**Goal:** evidence-based gate experiment + alternative attack on the
same-artist failure mode, without retraining.

**Hypotheses tested:**
- H1 (NMFP transfer): does Araz et al. ISMIR 2025 recipe transfer to Indian classical?
- H2 (hubness): is NAFP's same-artist failure mode a global hub problem that
  inference-time post-processing (Inverted Softmax, CSLS) can attack?

## Headline numbers (HR@1 on Saraga 1000 main queries)

| Cell | NAFP-ckpt-10 baseline | NMFP-ckpt-100 (Araz 2025) | NAFP+InvSoftmax | NAFP+CSLS |
|---|---|---|---|---|
| main_1s | 0.983 (1s) | 1.0000 | 0.9830 | 0.9830 |
| main_3s | 0.998 (3s) | 1.0000 | 0.9980 | 0.9980 |
| main_5s | 0.999 (5s) | 1.0000 | 0.9990 | 0.9980 |
| main_10s | 1.000 (10s) | 1.0000 | 1.0000 | 1.0000 |

## Failure-mode breakdown (NAFP vs NMFP same-artist misses)

### main_1s

- NAFP: HR@1=0.9830  n_miss=17  same-artist miss=14
- NMFP: HR@1=1.0000  n_miss=0  same-artist miss=0
- McNemar: b=0  c=17  Δhits=+17  p=0.0000
- Bonferroni-significant (α/8=0.00625): YES
- Queries fixed by NMFP: 17
- Queries broken by NMFP: 0

### main_3s

- NAFP: HR@1=0.9980  n_miss=2  same-artist miss=2
- NMFP: HR@1=1.0000  n_miss=0  same-artist miss=0
- McNemar: b=0  c=2  Δhits=+2  p=0.5000
- Bonferroni-significant (α/8=0.00625): no
- Queries fixed by NMFP: 2
- Queries broken by NMFP: 0

### main_5s

- NAFP: HR@1=0.9990  n_miss=1  same-artist miss=1
- NMFP: HR@1=1.0000  n_miss=0  same-artist miss=0
- McNemar: b=0  c=1  Δhits=+1  p=1.0000
- Bonferroni-significant (α/8=0.00625): no
- Queries fixed by NMFP: 1
- Queries broken by NMFP: 0

## What this means

Decision tree based on results:

- **If NMFP beats baseline AND mechanism is artist-hub:** add NMFP as system #5,
  cite Araz et al., write up as cross-domain benchmark contribution.
- **If NMFP does NOT beat baseline:** recipe doesn't transfer to non-Western tonal
  music. Publishable negative result. Focus remaining time on hubness post-proc.
- **If hubness (CSLS / InvSoftmax) beats baseline:** mechanism is geometry, not
  data. Publishable inference-time fix for non-Western music fingerprinting.

Detailed comparison JSON: `data/results/nafp/nmfp_eval/comparison/`
All intermediate logs: `data/results/nafp/nmfp_eval/overnight.log`