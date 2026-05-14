# Demo Clip Pack — Audio Fingerprinting for Indian Classical Music

**Live demo:** https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo

This pack contains **67 clips** organized into three categories:

| Category | Count | What it tests |
|---|---|---|
| `in_library` | 40 | the system correctly identifies known Saraga recordings |
| `out_of_library` | 22 | the system correctly rejects audio not in the Saraga library |
| `edge_case` | 5 | robustness to silence, noise, too-short clips, and amplitude/length extremes |

## How to use

1. Open the demo: https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo
2. Drag any `.wav` file from this folder into the Audio input.
3. Click **Identify**.
4. Compare against the `expected_verdict` column below.

All Saraga in-library clips should produce **MATCH** (most at high confidence).
All FMA out-of-library clips should produce **NO MATCH**.
Edge cases probe the system's failure handling — see notes per file.

## Full manifest

See `manifest.csv` for filename, expected verdict, source offset, and ground-truth metadata.

## File naming convention

- `saraga_in_library/NN_<corpus>_<length>_<query_id>.wav`
- `fma_out_of_library/NN_fma_<length>_<track_id>.wav`
- `edge_cases/NN_<descriptor>.wav`

## Licenses

- Saraga clips: CC-BY-NC-SA 4.0 (derivative of Saraga 1.5 audio)
- FMA clips: CC-BY 4.0 (derivative of FMA-medium)
- Edge cases (synthetic): public domain

## Reproducibility

Regenerable from `scripts/build_demo_clip_pack.py` (seed=20260515).
