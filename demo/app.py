"""Audio Fingerprinting for Indian Classical Music — Gradio demo.

Public demo for the Recipe v3 NAFP model (3-seed Bonferroni-significant
improvement on Saraga 1.5 1-second retrieval) with calibrated
out-of-library rejection.

Designed to run locally (Mac / Linux) and on Hugging Face Spaces (CPU).

PRE-LOADED AT STARTUP (no per-request cost):
- Recipe v3 seed 42 ckpt-30 encoder (frozen NAFP CNN, 128-D L2-normalized embeddings)
- FAISS IndexFlatIP over 690,414 Saraga ref segments
- Threshold table from data/results/threshold_calibration/threshold.json
- Saraga ref metadata (357 tracks: artist, raaga, taal, work)
"""
from __future__ import annotations

# MUST come before any TF import (silent partial-restore otherwise)
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Allow this app to run from repo root OR from a self-contained demo bundle.
# In the HF Spaces deployment we copy the upstream NAFP code under demo/upstream.
UPSTREAM_LOCAL = HERE / "upstream"
UPSTREAM_REPO = REPO / "scripts" / "nafp" / "upstream"
if UPSTREAM_LOCAL.exists():
    sys.path.insert(0, str(UPSTREAM_LOCAL))
elif UPSTREAM_REPO.exists():
    sys.path.insert(0, str(UPSTREAM_REPO))
else:
    raise RuntimeError("Cannot find NAFP upstream code at demo/upstream or scripts/nafp/upstream")

# Path map — try local artifacts first (for HF Spaces self-contained layout),
# fall back to repo-level paths (for local dev).
def _find(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {candidates} exist")


CKPT_INDEX = _find(
    HERE / "artifacts" / "ckpt-30.index",
    REPO / "data" / "results" / "nafp" / "recipe_v3_30ep" / "seed42" / "ckpt-30.index",
)
CKPT_PREFIX = str(CKPT_INDEX.with_suffix(""))  # tf wants the .data/.index sibling stem
CFG_PATH = _find(
    HERE / "artifacts" / "recipe_v3.yaml",
    REPO / "data" / "results" / "nafp" / "recipe_v3_30ep" / "recipe_v3.yaml",
)
REF_EMBS_PATH = _find(
    HERE / "artifacts" / "ref_embs.mm",
    REPO / "data" / "results" / "nafp" / "recipe_v3_30ep" / "seed42_eval" / "ref_embs.mm",
)
REF_LOOKUP_PATH = _find(
    HERE / "artifacts" / "ref_segment_lookup.parquet",
    REPO / "data" / "results" / "nafp" / "recipe_v3_30ep" / "seed42_eval" / "ref_segment_lookup.parquet",
)
THRESHOLD_PATH = _find(
    HERE / "artifacts" / "threshold.json",
    REPO / "data" / "results" / "threshold_calibration" / "threshold.json",
)
HI_REFS = _find(HERE / "artifacts" / "refs_hindustani.csv",
                REPO / "data" / "manifests" / "hindustani" / "refs.csv")
CA_REFS = _find(HERE / "artifacts" / "refs_carnatic.csv",
                REPO / "data" / "manifests" / "carnatic" / "refs.csv")


# ── Constants ─────────────────────────────────────────────────────────────
FS = 8000             # NAFP/recipe-v3 native sample rate
WIN = 8000            # 1-second window
HOP = 4000            # 0.5-second hop (matches NAFP training)
EMB_DIM = 128
N_REF_SEGS = 690_414  # fixed by recipe v3 seed 42 indexing run
MAX_QUERY_SEC = 30    # server-side max input duration to bound CPU + RAM


# ── Module-load (one-time) initialization ─────────────────────────────────
print("[startup] loading recipe v3 seed 42 encoder, FAISS index, thresholds...", flush=True)
_t0 = time.time()

import tensorflow as tf
import yaml

with open(CFG_PATH) as f:
    _CFG = yaml.safe_load(f)
assert int(_CFG["MODEL"]["FS"]) == FS, "config FS != 8000"
assert int(_CFG["MODEL"]["EMB_SZ"]) == EMB_DIM, "config EMB_SZ != 128"

tf.keras.backend.clear_session()
from model.generate import build_fp  # noqa: E402  (must come after sys.path edit)
M_PRE, M_FP = build_fp(_CFG)
_ = M_FP(M_PRE(tf.zeros((1, 1, WIN), dtype=tf.float32))).numpy()  # build sublayers
_ckpt = tf.train.Checkpoint(model=M_FP)
_ckpt.restore(CKPT_PREFIX).expect_partial()
print(f"[startup] encoder + ckpt-30 loaded ({time.time()-_t0:.1f}s)", flush=True)

LOOKUP = pd.read_parquet(REF_LOOKUP_PATH)
assert len(LOOKUP) == N_REF_SEGS, f"lookup has {len(LOOKUP)} rows, expected {N_REF_SEGS}"
LOOKUP_BY_IDX = LOOKUP.set_index("global_idx")
EMBS = np.memmap(REF_EMBS_PATH, dtype=np.float32, mode="r", shape=(N_REF_SEGS, EMB_DIM))
print(f"[startup] index lookup loaded ({time.time()-_t0:.1f}s, {N_REF_SEGS:,} segments)", flush=True)

import faiss  # noqa: E402
INDEX = faiss.IndexFlatIP(EMB_DIM)
INDEX.add(np.ascontiguousarray(EMBS, dtype=np.float32))
print(f"[startup] FAISS IndexFlatIP built ({time.time()-_t0:.1f}s)", flush=True)

# Ref metadata: ref_id -> {artist, raaga, taal, corpus, work}
_refs = pd.concat([pd.read_csv(HI_REFS), pd.read_csv(CA_REFS)], ignore_index=True)
REF_META = {
    r.ref_id: {
        "artist": r.artists if pd.notna(r.artists) else "—",
        "raaga": r.raagas if pd.notna(r.raagas) else "—",
        "taala": r.taalas if pd.notna(r.taalas) else "—",
        "corpus": r.corpus,
        "work": r.works if "works" in _refs.columns and pd.notna(r.works) else "—",
    }
    for _, r in _refs.iterrows()
}

with open(THRESHOLD_PATH) as f:
    _TH = json.load(f)
T_DEFAULT = float(_TH["T_default"])
T_HIGH = float(_TH["T_high_confidence"])
print(f"[startup] thresholds: T_default={T_DEFAULT:.4f}  T_high={T_HIGH:.4f} "
      f"({time.time()-_t0:.1f}s)", flush=True)
print(f"[startup] READY in {time.time()-_t0:.1f}s\n", flush=True)


# ── Inference pipeline ────────────────────────────────────────────────────
def _slice_segments(audio: np.ndarray) -> np.ndarray:
    """Cut overlapping 1-sec windows at 0.5-sec hop. Returns (n_seg, 1, 8000)."""
    n = audio.shape[0]
    if n < WIN:
        return np.empty((0, 1, WIN), dtype=np.float32)
    n_seg = (n - WIN) // HOP + 1
    out = np.empty((n_seg, 1, WIN), dtype=np.float32)
    for i in range(n_seg):
        out[i, 0, :] = audio[i * HOP:i * HOP + WIN]
    return out


def _encode(audio: np.ndarray, batch: int = 64) -> np.ndarray:
    """Slice → forward → L2-normalize. Returns (n_seg, 128)."""
    segs = _slice_segments(audio.astype(np.float32))
    if segs.shape[0] == 0:
        return np.empty((0, EMB_DIM), dtype=np.float32)
    out = np.empty((segs.shape[0], EMB_DIM), dtype=np.float32)
    for s in range(0, segs.shape[0], batch):
        e = min(s + batch, segs.shape[0])
        emb = M_FP(M_PRE(tf.constant(segs[s:e]))).numpy()
        out[s:e] = emb
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


def _sequence_search(q_emb: np.ndarray, k_probe: int = 20, top_k: int = 5):
    """Identical to scripts/nafp/recipe_v2_eval/eval.py:sequence_search."""
    L = q_emb.shape[0]
    if L == 0:
        return []
    _, I = INDEX.search(q_emb.astype(np.float32), k_probe)
    for off in range(L):
        I[off, :] -= off
    cands = np.unique(I[I >= 0])
    cands = cands[cands + L <= N_REF_SEGS]
    if len(cands) == 0:
        return []
    scores = np.empty(len(cands), dtype=np.float32)
    for ci, cid in enumerate(cands):
        chunk = np.asarray(EMBS[cid:cid + L], dtype=np.float32)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def _identify(audio_path: str) -> tuple[str, str, pd.DataFrame]:
    """Returns (verdict_md, summary_md, top5_df)."""
    if not audio_path:
        return ("**Upload or record a clip to start.**", "", pd.DataFrame())

    import librosa
    try:
        audio, _ = librosa.load(audio_path, sr=FS, mono=True)
    except Exception as exc:
        return (f"⚠️ **Could not decode audio:** `{exc}`", "", pd.DataFrame())

    dur = audio.shape[0] / FS
    if dur < 1.0:
        return ("⚠️ **Clip is shorter than 1 second.**\n"
                "Please provide at least 1 second of audio.",
                "", pd.DataFrame())
    if dur > MAX_QUERY_SEC:
        audio = audio[: int(MAX_QUERY_SEC * FS)]
        truncated = f"_(query truncated to {MAX_QUERY_SEC} s)_"
    else:
        truncated = ""

    t0 = time.time()
    q_emb = _encode(audio)
    if q_emb.shape[0] == 0:
        return ("⚠️ **No usable audio found.**", "", pd.DataFrame())
    hits = _sequence_search(q_emb, k_probe=20, top_k=5)
    enc_ms = (time.time() - t0) * 1000

    if not hits:
        return ("**No match in library** (no candidate segments returned).",
                f"Query duration: {dur:.2f} s · processed in {enc_ms:.0f} ms",
                pd.DataFrame())

    top1_score = hits[0][1]
    rows = []
    for rank, (cid, score) in enumerate(hits, 1):
        info = LOOKUP_BY_IDX.loc[cid]
        ref_id = str(info["ref_id"])
        meta = REF_META.get(ref_id, {"artist": "—", "raaga": "—", "taala": "—", "corpus": "—"})
        rows.append({
            "Rank": rank,
            "Score": round(score, 4),
            "Reference": ref_id,
            "Corpus": meta["corpus"].capitalize(),
            "Artist": meta["artist"],
            "Raaga": meta["raaga"],
            "Taala": meta["taala"],
            "Offset (s)": round(float(info["segment_start_sec"]), 2),
        })
    top5_df = pd.DataFrame(rows)

    # Verdict
    if top1_score >= T_HIGH:
        verdict = (f"## ✅ Match found (high confidence)\n\n"
                   f"**{rows[0]['Artist']}** — *{rows[0]['Raaga']}* / *{rows[0]['Taala']}*\n\n"
                   f"`{rows[0]['Reference']}` at **{rows[0]['Offset (s)']:.1f} s** · "
                   f"score **{top1_score:.4f}**")
    elif top1_score >= T_DEFAULT:
        verdict = (f"## 🟡 Match found\n\n"
                   f"**{rows[0]['Artist']}** — *{rows[0]['Raaga']}* / *{rows[0]['Taala']}*\n\n"
                   f"`{rows[0]['Reference']}` at **{rows[0]['Offset (s)']:.1f} s** · "
                   f"score **{top1_score:.4f}** _(below high-confidence threshold "
                   f"{T_HIGH:.3f}; verify manually)_")
    else:
        verdict = (f"## ❌ No match in library\n\n"
                   f"Top-1 score = **{top1_score:.4f}** is below the calibrated threshold "
                   f"**{T_DEFAULT:.4f}** (FPR ≤ 5% on FMA-medium out-of-library probes).\n\n"
                   f"This query is likely **not** one of the 357 Saraga reference recordings. "
                   f"Top-5 below are shown for transparency only — treat as noise.")

    summary = (f"**Query duration:** {dur:.2f} s {truncated} · "
               f"**Encode + search:** {enc_ms:.0f} ms · "
               f"**Segments used:** {q_emb.shape[0]}")

    return verdict, summary, top5_df


# ── Gradio UI ─────────────────────────────────────────────────────────────
import gradio as gr  # noqa: E402

DISCLAIMER = (
    "> ⚠️ **Closed-world fingerprinter.** This system retrieves matches from "
    "**357 Saraga 1.5 recordings** only (108 Hindustani + 249 Carnatic). "
    "Queries from outside this library (e.g., Bollywood, Western pop) will "
    "be flagged **\"No match\"** by the calibrated threshold "
    f"(T = {T_DEFAULT:.4f}, FPR ≤ 5% on FMA-medium probes). "
    "Phone-mic noise robustness is **not benchmarked**."
)

ABOUT_MD = f"""
# About this demo

## What it does
Identifies which of **357 Saraga 1.5** Indian classical music recordings a query
clip came from, plus the offset in seconds. Uses our **Recipe v3** model — a
NAFP (Chang et al., ICASSP 2021) encoder with two recipe modifications from
Araz et al. (ISMIR 2025), trained from scratch on FMA-medium for 30 epochs.

## What's running
- **Encoder:** Recipe v3 seed 42 (NAFP CNN, ~9 M params, 128-D L2-normalized embeddings)
- **Library:** 357 Saraga 1.5 refs = 690,414 1-second segments at 0.5-s hop
- **Index:** FAISS IndexFlatIP (exact inner-product = cosine on unit-norm vectors)
- **Sample rate:** 8 kHz mono (NAFP native)

## Why this model
- HR@1 on Saraga main_1s (1-second queries): **0.995** (3-seed mean) vs baseline NAFP-ckpt-10 at 0.983
- **~68 % miss-rate reduction** across the 8-cell benchmark, ~71 % on the hardest cell
- Pre-registered Bonferroni-significant (pooled McNemar p = 3.18 × 10⁻⁶)
- See: [hf.co/datasets/Tachyeon/audio-fingerprint-indian-bench](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench)

## How "No match in library" is decided
Threshold **T = {T_DEFAULT:.4f}** chosen via pre-registered calibration:
- 200 random FMA-medium probes → top-1 score distribution
- T = smallest value such that **FPR ≤ 5 %** (false-accept rate on out-of-library)
- At chosen T: **TPR = 99.09 %** (in-library queries correctly accepted)
- High-confidence threshold **T = {T_HIGH:.4f}** at **FPR ≤ 1 %** (TPR = 85.4 %)
- Calibration protocol: locked before measurement; see project repo.

## Honest limitations
- **357 refs is a tiny library.** A real Shazam indexes millions.
- **Shorter queries are first-N truncations** in the benchmark; demo accepts any length 1-30 s.
- **Phone-mic noise robustness not benchmarked** — clean-query results may not hold.
- **Same-artist confusion still present.** Of 5 mean recipe-v3 misses on main_1s, most are same-artist top-1 (different recording, same singer).
- **Out-of-library probe set is FMA-medium only** (Western music). Indian-classical-but-not-Saraga (e.g., other Bollywood / CompMusic) may produce different score distributions.

## Privacy
Audio you upload is processed in-memory for fingerprint extraction. We don't
log or retain it on this Space. Hugging Face's general data-handling policy applies
to all interactions with this Space — see [huggingface.co/privacy](https://huggingface.co/privacy).

## Licenses
- **Code in this Space:** MIT
- **Model weights:** MIT (we trained these on FMA-medium)
- **Reference embeddings:** derivative of Saraga 1.5 (CC-BY-NC-SA 4.0). The 354 MB
  `ref_embs.mm` is shipped under CC-BY-NC-SA 4.0 inheritance. **Non-commercial only.**

## Citations
```
Chang et al. 2021. Neural Audio Fingerprint for High-specific Audio Retrieval based on Contrastive Learning. ICASSP. arXiv:2010.11910
Araz et al. 2025.   Enhancing Neural Audio Fingerprint Robustness to Real-World Conditions. ISMIR. arXiv:2506.22661
Srinivasamurthy, Gulati, Repetto, Serra 2021. Saraga: Open Datasets for Research on Indian Art Music. Zenodo 10.5281/zenodo.4301737
```
"""


with gr.Blocks(title="AFP for Indian Classical Music") as demo:
    gr.Markdown("# Audio Fingerprinting for Indian Classical Music")
    gr.Markdown(DISCLAIMER)

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        format="wav",
                        label="Query audio (upload preferred; microphone experimental)",
                    )
                    submit_btn = gr.Button("Identify", variant="primary")
                    gr.Markdown(
                        "Tip: clips of **1-30 seconds** work; **upload** is more reliable "
                        "than mic recording across browsers (especially iOS Safari)."
                    )
                with gr.Column(scale=1):
                    verdict_md = gr.Markdown("**Upload or record a clip to start.**")
                    summary_md = gr.Markdown()

            gr.Markdown("### Top-5 candidates (transparency)")
            top5_df = gr.Dataframe(
                headers=["Rank", "Score", "Reference", "Corpus", "Artist", "Raaga", "Taala", "Offset (s)"],
                interactive=False,
                wrap=True,
            )

            submit_btn.click(_identify, inputs=audio_in, outputs=[verdict_md, summary_md, top5_df])

        with gr.Tab("About"):
            gr.Markdown(ABOUT_MD)

    gr.Markdown(
        "[📦 Dataset on Hugging Face](https://huggingface.co/datasets/Tachyeon/audio-fingerprint-indian-bench) · "
        "[💻 GitHub repo (private)](https://github.com/ipritamdash/afp-indian-classical) · "
        "Model: Recipe v3 seed 42 (NAFP + 2 NMFP recipe fixes, 30 ep × BSZ 320)"
    )


if __name__ == "__main__":
    # `share=False` is appropriate for HF Spaces (they handle public URL).
    # On local dev set share=True for a temporary public tunnel.
    demo.queue(default_concurrency_limit=2, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        show_error=True,
        theme=gr.themes.Soft(),
    )
