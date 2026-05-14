"""Out-of-library threshold calibration for recipe v3 seed 42.

Per PROTOCOL.md (locked 2026-05-15, before measurement):
  - 200 random FMA-medium clips streamed from fma_medium.zip
  - 1 random second per clip at fs=8000
  - Encoded via recipe v3 seed 42, FAISS-searched against existing Saraga index
  - Compared to in-library top-1 score distribution from main_1s correct matches
  - Threshold: smallest T such that FPR ≤ 0.05 on OOL set

Outputs to data/results/threshold_calibration/.
"""
from __future__ import annotations
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/results/threshold_calibration"
FMA_ZIP = REPO / "data/fma/fma_medium.zip"
CKPT_DIR = REPO / "data/results/nafp/recipe_v3_30ep/seed42"
CFG_PATH = REPO / "data/results/nafp/recipe_v3_30ep/recipe_v3.yaml"
REF_EMBS = REPO / "data/results/nafp/recipe_v3_30ep/seed42_eval/ref_embs.mm"
REF_LOOKUP = REPO / "data/results/nafp/recipe_v3_30ep/seed42_eval/ref_segment_lookup.parquet"
MAIN_1S_RESULTS = REPO / "data/results/nafp/recipe_v3_30ep/seed42_eval/main_1s/query_results.parquet"
HI_QUERIES = REPO / "data/manifests/hindustani/queries_1s.csv"
CA_QUERIES = REPO / "data/manifests/carnatic/queries_1s.csv"

SEED = 20260515
N_PROBES = 200
FS = 8000


def build_encoder():
    """Identical pattern to scripts/nafp/recipe_v2_eval/eval.py:build_encoder."""
    sys.path.insert(0, str(REPO / "scripts/nafp/upstream"))
    import tensorflow as tf
    import yaml
    tf.keras.backend.clear_session()
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    print(f"[cfg] F_MIN={cfg['MODEL']['F_MIN']}  EMB_SZ={cfg['MODEL']['EMB_SZ']}", flush=True)
    from model.generate import build_fp
    m_pre, m_fp = build_fp(cfg)
    _ = m_fp(m_pre(tf.zeros((1, 1, FS), dtype=tf.float32))).numpy()
    ckpt = tf.train.Checkpoint(model=m_fp)
    ckpt.restore(str(CKPT_DIR / "ckpt-30")).expect_partial()
    print(f"[ckpt] Restored ckpt-30 from {CKPT_DIR}", flush=True)
    return m_pre, m_fp


def encode_1sec(audio_1d: np.ndarray, m_pre, m_fp) -> np.ndarray:
    """Encode exactly one 1-second window at fs=8000 (8000 samples → (1,1,8000) input).
    Returns L2-normalized (1, 128) embedding."""
    import tensorflow as tf
    assert audio_1d.shape == (FS,), f"expected ({FS},), got {audio_1d.shape}"
    seg = audio_1d.reshape(1, 1, FS).astype(np.float32)
    emb = m_fp(m_pre(tf.constant(seg))).numpy()
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def sequence_search_topk(q_emb: np.ndarray, index, embs: np.memmap,
                         *, k_probe: int = 20, top_k: int = 5):
    """Same as scripts/nafp/recipe_v2_eval/eval.py:sequence_search.
    For a 1-second query, L=1 (single segment). Returns [(cand_idx, score)]."""
    L = q_emb.shape[0]
    _, I = index.search(q_emb.astype(np.float32), k_probe)
    for off in range(L):
        I[off, :] -= off
    cands = np.unique(I[I >= 0])
    cands = cands[cands + L <= embs.shape[0]]
    if len(cands) == 0:
        return []
    scores = np.empty(len(cands), dtype=np.float32)
    for ci, cid in enumerate(cands):
        chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def sample_fma_paths(rng: np.random.Generator) -> list[str]:
    """List MP3 paths in fma_medium.zip and pick N_PROBES with seeded RNG."""
    with zipfile.ZipFile(FMA_ZIP) as zf:
        all_mp3 = [n for n in zf.namelist() if n.endswith(".mp3")]
    print(f"[fma] total MP3s in zip: {len(all_mp3):,}", flush=True)
    idx = rng.choice(len(all_mp3), size=N_PROBES, replace=False)
    return [all_mp3[i] for i in sorted(idx)]


def stream_decode(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, int]:
    """Read MP3 bytes from zip → librosa decode → mono fs=8000."""
    import librosa
    data = zf.read(member)
    # librosa.load needs a file-like or path; soundfile/audioread accept BytesIO
    audio, sr = librosa.load(io.BytesIO(data), sr=FS, mono=True)
    return audio.astype(np.float32, copy=False), sr


def measure_ool_scores(m_pre, m_fp, index, embs, lookup_by_idx, rng) -> pd.DataFrame:
    fma_paths = sample_fma_paths(rng)
    rows = []
    t0 = time.time()
    with zipfile.ZipFile(FMA_ZIP) as zf:
        for i, mp3 in enumerate(fma_paths):
            try:
                audio, _ = stream_decode(zf, mp3)
            except Exception as exc:
                print(f"  [skip] {mp3}: decode failed {exc}", flush=True)
                continue
            if audio.shape[0] < FS:
                continue
            max_off = audio.shape[0] - FS
            off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
            clip = audio[off:off + FS]
            q_emb = encode_1sec(clip, m_pre, m_fp)
            hits = sequence_search_topk(q_emb, index, embs, k_probe=20, top_k=5)
            if not hits:
                continue
            top_cid, top_score = hits[0]
            info = lookup_by_idx.loc[top_cid]
            rows.append({
                "fma_path": mp3,
                "fma_offset_sec": off / FS,
                "top1_score": top_score,
                "top1_ref_id": str(info["ref_id"]),
                "top1_segment_idx": int(info["segment_idx"]),
            })
            if (i + 1) % 20 == 0:
                print(f"[ool] {i+1}/{N_PROBES} elapsed={time.time()-t0:.0f}s "
                      f"latest_score={top_score:.4f}", flush=True)
    return pd.DataFrame(rows)


def load_in_library_scores() -> np.ndarray:
    """Read main_1s eval parquet; return top-1 scores for CORRECT matches only."""
    queries = pd.concat([
        pd.read_csv(HI_QUERIES), pd.read_csv(CA_QUERIES)
    ], ignore_index=True)
    truth = dict(zip(queries.query_id, queries.ref_id))
    df = pd.read_parquet(MAIN_1S_RESULTS)
    top1 = df[df["rank"] == 1].copy()
    top1["truth"] = top1["query_id"].map(truth)
    top1["correct"] = top1["predicted_ref_id"] == top1["truth"]
    return top1.loc[top1["correct"], "nafp_score"].to_numpy(dtype=np.float64)


def pick_threshold(in_scores: np.ndarray, ool_scores: np.ndarray, target_fpr: float):
    """Smallest T such that FPR_on_OOL = mean(ool_scores >= T) <= target_fpr."""
    sorted_ool = np.sort(ool_scores)[::-1]  # descending
    n_ool = len(sorted_ool)
    max_false_accepts = int(np.floor(target_fpr * n_ool))
    if max_false_accepts == 0:
        T = max(sorted_ool[0] + 1e-6, 1e-6)
    else:
        T = sorted_ool[max_false_accepts - 1] + 1e-6
    fpr = float(np.mean(ool_scores >= T))
    tpr = float(np.mean(in_scores >= T))
    return float(T), tpr, fpr


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2/n
    c = (p + z**2/(2*n))/denom
    h = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)))/denom
    return (max(0, c-h), min(1, c+h))


def plot_results(in_scores, ool_scores, T_05, T_01, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histograms
    ax = axes[0]
    bins = np.linspace(0, 1, 61)
    ax.hist(ool_scores, bins=bins, alpha=0.6, color="#c0392b", label=f"Out-of-library (FMA, n={len(ool_scores)})", density=True)
    ax.hist(in_scores, bins=bins, alpha=0.6, color="#27ae60", label=f"In-library correct matches (Saraga main_1s, n={len(in_scores)})", density=True)
    ax.axvline(T_05, color="#2c3e50", linestyle="--", label=f"T@FPR≤5% = {T_05:.3f}")
    ax.axvline(T_01, color="#34495e", linestyle=":", label=f"T@FPR≤1% = {T_01:.3f}")
    ax.set_xlabel("Top-1 sequence-similarity score")
    ax.set_ylabel("Density")
    ax.set_title("Score distributions: in-library vs out-of-library")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1)

    # ROC
    ax = axes[1]
    thresholds = np.linspace(0, 1, 1001)
    tpr_vals = [np.mean(in_scores >= T) for T in thresholds]
    fpr_vals = [np.mean(ool_scores >= T) for T in thresholds]
    ax.plot(fpr_vals, tpr_vals, color="#2c3e50")
    ax.scatter([np.mean(ool_scores >= T_05)], [np.mean(in_scores >= T_05)],
               color="#c0392b", s=80, zorder=5, label=f"T={T_05:.3f} (FPR≤5%)")
    ax.scatter([np.mean(ool_scores >= T_01)], [np.mean(in_scores >= T_01)],
               color="#8e44ad", s=80, zorder=5, label=f"T={T_01:.3f} (FPR≤1%)")
    ax.plot([0, 1], [0, 1], color="#bbb", linestyle="--", alpha=0.5)
    ax.set_xlabel("False-positive rate (OOL accepted as match)")
    ax.set_ylabel("True-positive rate (in-library correctly accepted)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "histograms_and_roc.png", dpi=140)
    print(f"[plot] wrote {out_dir / 'histograms_and_roc.png'}", flush=True)


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[step 1] load encoder", flush=True)
    m_pre, m_fp = build_encoder()

    print("\n[step 2] load Saraga ref index + lookup", flush=True)
    lookup = pd.read_parquet(REF_LOOKUP)
    n_segs = len(lookup)
    embs = np.memmap(REF_EMBS, dtype=np.float32, mode="r", shape=(n_segs, 128))
    print(f"  {n_segs:,} ref segments", flush=True)

    print("\n[step 3] build FAISS IndexFlatIP", flush=True)
    import faiss
    index = faiss.IndexFlatIP(128)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))

    print("\n[step 4] measure OOL scores on 200 FMA-medium probes", flush=True)
    lookup_by_idx = lookup.set_index("global_idx")
    ool_df = measure_ool_scores(m_pre, m_fp, index, embs, lookup_by_idx, rng)
    ool_df.to_csv(OUT_DIR / "ool_scores.csv", index=False)
    print(f"[ool] saved {len(ool_df)} rows to ool_scores.csv", flush=True)

    print("\n[step 5] load in-library distribution", flush=True)
    in_scores = load_in_library_scores()
    ool_scores = ool_df["top1_score"].to_numpy()
    print(f"  in-library: n={len(in_scores)}, mean={in_scores.mean():.4f}, "
          f"min={in_scores.min():.4f}, p5={np.percentile(in_scores, 5):.4f}", flush=True)
    print(f"  out-of-library: n={len(ool_scores)}, mean={ool_scores.mean():.4f}, "
          f"max={ool_scores.max():.4f}, p95={np.percentile(ool_scores, 95):.4f}", flush=True)

    print("\n[step 6] pick thresholds (pre-registered: FPR ≤ 0.05 default, FPR ≤ 0.01 high-conf)", flush=True)
    T_05, tpr_05, fpr_05 = pick_threshold(in_scores, ool_scores, target_fpr=0.05)
    T_01, tpr_01, fpr_01 = pick_threshold(in_scores, ool_scores, target_fpr=0.01)
    print(f"  T@FPR≤5%  = {T_05:.4f}  → actual FPR={fpr_05:.4f}, TPR={tpr_05:.4f}", flush=True)
    print(f"  T@FPR≤1%  = {T_01:.4f}  → actual FPR={fpr_01:.4f}, TPR={tpr_01:.4f}", flush=True)

    feasible = (tpr_05 >= 0.95)
    print(f"\n[feasibility] TPR @ FPR≤5% = {tpr_05:.4f}  → "
          f"{'FEASIBLE (pre-reg criterion met)' if feasible else 'INFEASIBLE'}", flush=True)

    # Wilson CIs
    ci_tpr_05 = wilson_ci(int(np.sum(in_scores >= T_05)), len(in_scores))
    ci_fpr_05 = wilson_ci(int(np.sum(ool_scores >= T_05)), len(ool_scores))
    ci_tpr_01 = wilson_ci(int(np.sum(in_scores >= T_01)), len(in_scores))
    ci_fpr_01 = wilson_ci(int(np.sum(ool_scores >= T_01)), len(ool_scores))

    print("\n[step 7] plots", flush=True)
    plot_results(in_scores, ool_scores, T_05, T_01, OUT_DIR)

    print("\n[step 8] write threshold.json + RESULTS.md", flush=True)
    thresholds = {
        "T_default": T_05,
        "T_high_confidence": T_01,
        "fpr_target_default": 0.05,
        "fpr_target_high_conf": 0.01,
        "tpr_at_default": tpr_05,
        "tpr_at_high_conf": tpr_01,
        "actual_fpr_default": fpr_05,
        "actual_fpr_high_conf": fpr_01,
        "n_in_library": len(in_scores),
        "n_out_of_library": len(ool_scores),
        "feasible": feasible,
        "ckpt": "recipe_v3 seed 42 ckpt-30",
        "seed": SEED,
    }
    (OUT_DIR / "threshold.json").write_text(json.dumps(thresholds, indent=2))

    md = []
    md.append("# Threshold Calibration Results — Recipe v3 seed 42\n")
    md.append("**Pre-registered:** see [`PROTOCOL.md`](PROTOCOL.md) (locked before measurement).")
    md.append("**Generated:** 2026-05-15.\n")
    md.append("## Score distributions\n")
    md.append("| Distribution | n | mean | min | p5 / p95 | max |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| In-library correct matches (Saraga main_1s) | {len(in_scores)} | {in_scores.mean():.4f} | {in_scores.min():.4f} | p5={np.percentile(in_scores, 5):.4f} | {in_scores.max():.4f} |")
    md.append(f"| Out-of-library (FMA-medium random) | {len(ool_scores)} | {ool_scores.mean():.4f} | {ool_scores.min():.4f} | p95={np.percentile(ool_scores, 95):.4f} | {ool_scores.max():.4f} |")
    md.append("")
    md.append("## Chosen thresholds (pre-registered rule)\n")
    md.append("| Operating point | T | FPR (OOL accepted) | TPR (in-library accepted) | TPR 95% CI |")
    md.append("|---|---|---|---|---|")
    md.append(f"| Default (FPR ≤ 0.05) | **{T_05:.4f}** | {fpr_05:.4f} (95% CI: {ci_fpr_05[0]:.3f}, {ci_fpr_05[1]:.3f}) | {tpr_05:.4f} | ({ci_tpr_05[0]:.3f}, {ci_tpr_05[1]:.3f}) |")
    md.append(f"| High-confidence (FPR ≤ 0.01) | **{T_01:.4f}** | {fpr_01:.4f} (95% CI: {ci_fpr_01[0]:.3f}, {ci_fpr_01[1]:.3f}) | {tpr_01:.4f} | ({ci_tpr_01[0]:.3f}, {ci_tpr_01[1]:.3f}) |")
    md.append("")
    md.append("## Feasibility check\n")
    md.append(f"- Pre-registered criterion: TPR ≥ 0.95 at FPR ≤ 0.05 → **{'PASS' if feasible else 'FAIL'}** (actual TPR = {tpr_05:.4f})")
    md.append("")
    md.append("## Use in the demo\n")
    md.append(f"- If top-1 score ≥ **{T_05:.4f}**: display `\"Match found\"` with the predicted ref")
    md.append(f"- If top-1 score ≥ **{T_01:.4f}**: also show a `\"high confidence\"` badge")
    md.append(f"- Else: display `\"No match in library (query may be out-of-scope)\"`")
    md.append("")
    md.append("## Visual\n")
    md.append("![Score histograms and ROC](histograms_and_roc.png)\n")
    (OUT_DIR / "RESULTS.md").write_text("\n".join(md))
    print(f"[done] wrote {OUT_DIR / 'RESULTS.md'} and threshold.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
