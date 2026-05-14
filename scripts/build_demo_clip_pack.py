"""Build a curated 50+ clip pack for the live demo at
https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo

Output: demo_clips/ with subdirs + README.md + manifest.csv

Composition (50 clips total):
  30 × Saraga in-library  (15 Hindustani + 15 Carnatic, varied artists/lengths)
  15 × FMA out-of-library (varied lengths 1-30s)
   5 × Edge cases (silent, noise, sub-1s, very long Saraga, low-volume)
"""
from __future__ import annotations
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "demo_clips"
SARAGA_SUBDIR = OUT / "saraga_in_library"
FMA_SUBDIR = OUT / "fma_out_of_library"
EDGE_SUBDIR = OUT / "edge_cases"
FS = 8000


def clean_out():
    if OUT.exists():
        shutil.rmtree(OUT)
    SARAGA_SUBDIR.mkdir(parents=True)
    FMA_SUBDIR.mkdir(parents=True)
    EDGE_SUBDIR.mkdir(parents=True)


def pick_saraga_clips(rng: np.random.Generator, n_per_corpus: int = 20):
    """40 Saraga clips: 20 H + 20 C across 1s/3s/5s/10s, varied by ref.
    Sample directly from the existing queries — most diverse, no artist constraint."""
    lengths = ["1s", "3s", "5s", "10s"]
    rows = []
    for corpus in ["hindustani", "carnatic"]:
        # Pool all queries across all lengths, sample uniformly
        all_q = []
        for L in lengths:
            fname = "queries.csv" if L == "10s" else f"queries_{L}.csv"
            df = pd.read_csv(REPO / f"data/manifests/{corpus}/{fname}")
            df["_length"] = L
            all_q.append(df)
        pool = pd.concat(all_q, ignore_index=True)
        # Sample n_per_corpus with seed, deduplicate by (ref_id, length) for variety
        sampled = pool.sample(n=n_per_corpus * 3, random_state=rng.integers(0, 2**31))
        seen, picks = set(), []
        for _, q in sampled.iterrows():
            key = (q.ref_id, q._length)
            if key in seen: continue
            seen.add(key)
            picks.append(q)
            if len(picks) >= n_per_corpus: break
        print(f"  {corpus:12s}: {len(picks)} unique (ref × length) picks")
        for q in picks:
            rows.append({
                "category": "in_library",
                "corpus": corpus, "length": q._length,
                "src_query_id": q.query_id, "src_path": q.audio_path,
                "ref_id": q.ref_id, "offset_sec": q.offset_sec,
            })
    return rows


def pick_fma_clips(rng: np.random.Generator, n: int = 25):
    """25 FMA OOL clips with varied lengths (oversampled — some may drop)."""
    ool = pd.read_csv(REPO / "data/results/threshold_calibration/ool_scores.csv")
    # Length distribution: balance across 1/3/5/10/15/30 sec
    length_dist = [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 20, 30, 30, 30][:n]
    picks = ool.sample(n=min(n, len(ool)), random_state=rng.integers(0, 2**31))
    return [
        {"category": "out_of_library",
         "src_path_in_zip": row["fma_path"],
         "length_sec": L,
         "src_track_id": Path(row["fma_path"]).stem}
        for (_, row), L in zip(picks.iterrows(), length_dist)
    ]


def build_saraga_clips(rng):
    rows = pick_saraga_clips(rng)
    manifest = []
    for i, r in enumerate(rows, 1):
        # Copy the existing query WAV (already at fs=8000)
        src = Path(r["src_path"])
        if not src.exists():
            print(f"  [skip] missing {src}")
            continue
        out_name = f"{i:02d}_{r['corpus']}_{r['length']}_{r['src_query_id']}.wav"
        out_path = SARAGA_SUBDIR / out_name
        shutil.copy(src, out_path)
        manifest.append({
            "filename": f"saraga_in_library/{out_name}",
            "category": "in_library",
            "corpus": r["corpus"],
            "length": r["length"],
            "truth_ref_id": r["ref_id"],
            "source_offset_sec": float(r["offset_sec"]),
            "expected_verdict": "MATCH or HIGH (in library)",
            "notes": f"Saraga query cut, length={r['length']}",
        })
    print(f"  wrote {len(manifest)} Saraga clips")
    return manifest


def build_fma_clips(rng):
    rows = pick_fma_clips(rng)
    manifest = []
    import librosa
    with zipfile.ZipFile(REPO / "data/fma/fma_medium.zip") as zf:
        for i, r in enumerate(rows, 1):
            mp3 = r["src_path_in_zip"]
            try:
                data = zf.read(mp3)
                audio, _ = librosa.load(io.BytesIO(data), sr=FS, mono=True)
            except Exception as exc:
                print(f"  [skip] {mp3}: {exc}")
                continue
            L_samples = r["length_sec"] * FS
            if audio.shape[0] < L_samples + FS:
                continue
            max_off = audio.shape[0] - L_samples
            off = int(rng.integers(0, max_off + 1))
            clip = audio[off:off + L_samples].astype(np.float32)
            out_name = f"{i:02d}_fma_{r['length_sec']:02d}s_{r['src_track_id']}.wav"
            out_path = FMA_SUBDIR / out_name
            sf.write(out_path, clip, FS)
            manifest.append({
                "filename": f"fma_out_of_library/{out_name}",
                "category": "out_of_library",
                "corpus": "fma_medium",
                "length": f"{r['length_sec']}s",
                "truth_ref_id": "(not in library)",
                "source_offset_sec": off / FS,
                "expected_verdict": "NO MATCH (out of library)",
                "notes": f"FMA-medium random clip, track={r['src_track_id']}",
            })
    print(f"  wrote {len(manifest)} FMA clips")
    return manifest


def build_edge_clips(rng):
    """5 synthetic / corner-case clips."""
    cases = []
    # 1) Silent 5s
    sf.write(EDGE_SUBDIR / "01_silent_5s.wav", np.zeros(FS * 5, dtype=np.float32), FS)
    cases.append({"filename": "edge_cases/01_silent_5s.wav", "category": "edge_case",
                  "corpus": "synthetic", "length": "5s", "truth_ref_id": "(silence)",
                  "source_offset_sec": 0.0,
                  "expected_verdict": "NO MATCH (silence, low embedding similarity)",
                  "notes": "All zeros at fs=8000 for 5 seconds"})
    # 2) White noise 5s
    sf.write(EDGE_SUBDIR / "02_white_noise_5s.wav",
             (rng.standard_normal(FS * 5).astype(np.float32) * 0.1), FS)
    cases.append({"filename": "edge_cases/02_white_noise_5s.wav", "category": "edge_case",
                  "corpus": "synthetic", "length": "5s", "truth_ref_id": "(noise)",
                  "source_offset_sec": 0.0,
                  "expected_verdict": "NO MATCH (white noise)",
                  "notes": "Gaussian noise σ=0.1, fs=8000, 5 sec"})
    # 3) Too-short 0.5s (should be rejected by demo)
    sf.write(EDGE_SUBDIR / "03_too_short_0.5s.wav",
             (rng.standard_normal(FS // 2).astype(np.float32) * 0.1), FS)
    cases.append({"filename": "edge_cases/03_too_short_0.5s.wav", "category": "edge_case",
                  "corpus": "synthetic", "length": "0.5s", "truth_ref_id": "(too short)",
                  "source_offset_sec": 0.0,
                  "expected_verdict": "REJECTED (clip < 1 sec, demo error)",
                  "notes": "0.5 sec of noise; below 1-sec window — demo errors out"})
    # 4) Quiet (low-volume) Saraga 5s
    q5 = pd.read_csv(REPO / "data/manifests/hindustani/queries_5s.csv").iloc[10]
    import librosa
    a, _ = librosa.load(q5.audio_path, sr=FS, mono=True)
    a = a[:FS * 5] * 0.02   # 30dB quieter than normal
    sf.write(EDGE_SUBDIR / "04_saraga_quiet_5s.wav", a.astype(np.float32), FS)
    cases.append({"filename": "edge_cases/04_saraga_quiet_5s.wav", "category": "edge_case",
                  "corpus": "hindustani", "length": "5s", "truth_ref_id": q5.ref_id,
                  "source_offset_sec": float(q5.offset_sec),
                  "expected_verdict": "MATCH (model is invariant to amplitude)",
                  "notes": f"Saraga {q5.query_id} attenuated to 2%; tests amplitude invariance"})
    # 5) 30-second long Saraga (multiple segments — tests sequence aggregation)
    q10 = pd.read_csv(REPO / "data/manifests/hindustani/queries.csv").iloc[20]
    a, _ = librosa.load(q10.audio_path, sr=FS, mono=True)
    a = a[:FS * 10]  # 10 sec available; tile to ~30 sec
    if a.shape[0] < FS * 30:
        a = np.tile(a, int(np.ceil((FS * 30) / a.shape[0])))[:FS * 30]
    sf.write(EDGE_SUBDIR / "05_saraga_long_30s.wav", a.astype(np.float32), FS)
    cases.append({"filename": "edge_cases/05_saraga_long_30s.wav", "category": "edge_case",
                  "corpus": "hindustani", "length": "30s",
                  "truth_ref_id": q10.ref_id, "source_offset_sec": float(q10.offset_sec),
                  "expected_verdict": "MATCH (long-form sequence aggregation)",
                  "notes": f"Saraga {q10.query_id} tiled to 30 sec; max length demo accepts"})
    print(f"  wrote {len(cases)} edge-case clips")
    return cases


def write_readme(manifest_df: pd.DataFrame):
    md = []
    md.append("# Demo Clip Pack — Audio Fingerprinting for Indian Classical Music\n")
    md.append("**Live demo:** https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo\n")
    md.append(f"This pack contains **{len(manifest_df)} clips** organized into three categories:\n")
    md.append("| Category | Count | What it tests |")
    md.append("|---|---|---|")
    for cat, expected in [
        ("in_library",    "the system correctly identifies known Saraga recordings"),
        ("out_of_library", "the system correctly rejects audio not in the Saraga library"),
        ("edge_case",     "robustness to silence, noise, too-short clips, and amplitude/length extremes"),
    ]:
        n = (manifest_df["category"] == cat).sum()
        md.append(f"| `{cat}` | {n} | {expected} |")
    md.append("\n## How to use\n")
    md.append("1. Open the demo: https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo")
    md.append("2. Drag any `.wav` file from this folder into the Audio input.")
    md.append("3. Click **Identify**.")
    md.append("4. Compare against the `expected_verdict` column below.")
    md.append("\nAll Saraga in-library clips should produce **MATCH** (most at high confidence).")
    md.append("All FMA out-of-library clips should produce **NO MATCH**.")
    md.append("Edge cases probe the system's failure handling — see notes per file.\n")
    md.append("## Full manifest\n")
    md.append("See `manifest.csv` for filename, expected verdict, source offset, and ground-truth metadata.\n")
    md.append("## File naming convention\n")
    md.append("- `saraga_in_library/NN_<corpus>_<length>_<query_id>.wav`")
    md.append("- `fma_out_of_library/NN_fma_<length>_<track_id>.wav`")
    md.append("- `edge_cases/NN_<descriptor>.wav`\n")
    md.append("## Licenses\n")
    md.append("- Saraga clips: CC-BY-NC-SA 4.0 (derivative of Saraga 1.5 audio)")
    md.append("- FMA clips: CC-BY 4.0 (derivative of FMA-medium)")
    md.append("- Edge cases (synthetic): public domain\n")
    md.append(f"## Reproducibility\n\nRegenerable from `scripts/build_demo_clip_pack.py` (seed=20260515).\n")
    (OUT / "README.md").write_text("\n".join(md))
    print(f"  wrote README.md ({len(manifest_df)} rows in manifest)")


def main():
    print(f"[start] building demo clip pack at {OUT}")
    clean_out()
    rng = np.random.default_rng(20260515)
    print("\n[1/4] Saraga in-library clips")
    saraga = build_saraga_clips(rng)
    print("\n[2/4] FMA out-of-library clips")
    fma = build_fma_clips(rng)
    print("\n[3/4] Edge-case clips")
    edge = build_edge_clips(rng)
    print("\n[4/4] manifest + README")
    manifest = pd.DataFrame(saraga + fma + edge)
    manifest.to_csv(OUT / "manifest.csv", index=False)
    write_readme(manifest)

    print(f"\n[done] {len(manifest)} clips at {OUT}")
    print(f"  saraga_in_library:  {(manifest.category == 'in_library').sum()}")
    print(f"  fma_out_of_library: {(manifest.category == 'out_of_library').sum()}")
    print(f"  edge_cases:         {(manifest.category == 'edge_case').sum()}")
    # Summary table by length
    by_len = manifest.groupby(['category', 'length']).size().unstack(fill_value=0)
    print(f"\n{by_len}")


if __name__ == "__main__":
    sys.exit(main())
