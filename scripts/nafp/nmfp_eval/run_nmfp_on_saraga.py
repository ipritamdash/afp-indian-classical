"""Move 1: Run NMFP-published weights (ckpt-100, triplet variant, F_MIN=160) on Saraga.

GATE EXPERIMENT — answers: does Araz et al. ISMIR 2025 recipe transfer to Indian classical clean retrieval?

Sources verified:
- Zenodo 15719945: nmfp-triplet.zip, 172.9 MB, AGPLv3, ckpt-100
- Architecture identical to NAFP except cfg.MODEL.INPUT.F_MIN = 160 (vs NAFP's 300)
- Same CNN code path (model/fp/nnfp.py); same EMB_SZ=128

This script:
1. Builds NAFP encoder with NMFP's config (F_MIN=160)
2. Loads NMFP ckpt-100 weights with TF_USE_LEGACY_KERAS=1
3. Sanity-check: encode a known Saraga ref segment, verify embedding is non-trivial
4. Encodes all 357 Saraga refs → nmfp_ref_embs.mm
5. Builds FAISS IndexFlatIP, runs sequence-search on main_1s queries
6. Compares HR@1 to ckpt-10 baseline (0.983)

Output dir: data/results/nafp/nmfp_eval/saraga_main_1s/
"""
from __future__ import annotations

# CRITICAL: must be set BEFORE any TF import
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
NMFP_DIR = REPO / "data/nmfp/nmfp-triplet"
OUT_DIR = REPO / "data/results/nafp/nmfp_eval"


def setup_log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(__name__)


def build_cfg_for_nmfp() -> dict:
    """Build a NAFP-compatible cfg that mirrors NMFP's MODEL section.
    NMFP's config has different keys (MODEL.INPUT.F_MIN vs NAFP's MODEL.F_MIN);
    we need to translate to NAFP's flat MODEL section format."""
    import yaml
    with open(NMFP_DIR / "config.yaml") as f:
        nmfp = yaml.safe_load(f)
    m = nmfp["MODEL"]
    # NAFP build_fp expects flat keys under MODEL
    cfg = {
        "MODEL": {
            "FEAT": "melspec",
            "FS": int(m["AUDIO"]["FS"]),
            "DUR": float(m["AUDIO"]["SEGMENT_DUR"]),
            "HOP": 0.5,  # NAFP eval convention; query slicing only
            "STFT_WIN": int(m["INPUT"]["STFT_WIN"]),
            "STFT_HOP": int(m["INPUT"]["STFT_HOP"]),
            "F_MIN": float(m["INPUT"]["F_MIN"]),  # 160.0 from NMFP
            "F_MAX": float(m["INPUT"]["F_MAX"]),
            "N_MELS": int(m["INPUT"]["N_MELS"]),
            "EMB_SZ": int(m["ARCHITECTURE"]["EMB_SZ"]),
            "BN": str(m["ARCHITECTURE"]["BN"]),
        }
    }
    return cfg


def build_and_load_encoder(log: logging.Logger):
    """Build NAFP encoder with NMFP cfg, load NMFP ckpt-100, sanity check."""
    upstream = REPO / "scripts/nafp/upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    import tensorflow as tf
    tf.keras.backend.clear_session()
    log.info(f"[tf] keras module: {tf.keras.__name__}")

    cfg = build_cfg_for_nmfp()
    log.info(f"[cfg] F_MIN={cfg['MODEL']['F_MIN']} (NMFP)  vs NAFP F_MIN=300")

    from model.generate import build_fp
    m_pre, m_fp = build_fp(cfg)
    # Build sublayers via warmup BEFORE attempting restore
    _ = m_fp(m_pre(tf.zeros((1, 1, 8000), dtype=tf.float32))).numpy()

    # Manual restore so we can inspect partial-restore status
    ckpt_path = str(NMFP_DIR / "ckpt-100")
    checkpoint = tf.train.Checkpoint(model=m_fp)
    status = checkpoint.restore(ckpt_path)
    # NMFP ckpt has optimizer state we don't need; use expect_partial then check
    status.expect_partial()
    log.info(f"[ckpt] Restored from {ckpt_path}")

    # Diagnostic: check that MODEL variables were restored (not just optimizer)
    # We'll do this via the empirical sanity check below.
    return m_pre, m_fp, cfg


def sanity_check_encoder(m_pre, m_fp, log: logging.Logger) -> bool:
    """Encode the same Saraga ref segment we used for the NAFP sanity check.
    NMFP weights are DIFFERENT from NAFP ckpt-10 (different recipe), so we don't
    expect cos≈1.0 against the stored Kaggle embedding. We expect:
      (a) Output embedding is L2-normalizable to unit norm
      (b) Two different inputs produce different embeddings (model is not constant)
      (c) Same input twice produces identical embedding (model is deterministic)
    """
    import tensorflow as tf
    import librosa

    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)
    ref = refs[refs.ref_id == "hindustani_0_Raag_Shree"].iloc[0]
    audio, _ = librosa.load(ref["audio_path"], sr=8000, mono=True)

    # Two different segments
    seg_a = audio[1028 * 4000:1028 * 4000 + 8000].reshape(1, 1, 8000).astype(np.float32)
    seg_b = audio[2000 * 4000:2000 * 4000 + 8000].reshape(1, 1, 8000).astype(np.float32)

    emb_a1 = m_fp(m_pre(tf.constant(seg_a))).numpy()
    emb_a2 = m_fp(m_pre(tf.constant(seg_a))).numpy()  # deterministic test
    emb_b = m_fp(m_pre(tf.constant(seg_b))).numpy()

    norm_a = float(np.linalg.norm(emb_a1[0]))
    norm_b = float(np.linalg.norm(emb_b[0]))
    a_a_cos = float((emb_a1[0] / norm_a) @ (emb_a2[0] / np.linalg.norm(emb_a2[0])))
    a_b_cos = float((emb_a1[0] / norm_a) @ (emb_b[0] / norm_b))

    log.info(f"[sanity] norm(emb_seg_1028) = {norm_a:.4f}  norm(emb_seg_2000) = {norm_b:.4f}")
    log.info(f"[sanity] determinism cos(a,a) = {a_a_cos:.6f}  (must be ≈ 1.0)")
    log.info(f"[sanity] discriminability cos(a,b) = {a_b_cos:.4f}  (must be < 0.99 for non-trivial model)")

    # Also compare against NAFP-ckpt-10's embedding for the same segment.
    # This should NOT be ~1.0 (different model) and NOT be ~0 (totally orthogonal).
    embs_nafp = np.memmap(
        REPO / "data/results/nafp/saraga_only_main/ref_embs.mm",
        dtype=np.float32, mode="r", shape=(690414, 128))
    nafp_stored = embs_nafp[1028]
    nmfp_vs_nafp_cos = float((emb_a1[0] / norm_a) @ nafp_stored)
    log.info(f"[sanity] cos(nmfp_mac, nafp_kaggle_stored) = {nmfp_vs_nafp_cos:.4f}  "
             f"(expect non-zero — same audio, different model)")

    # Pass conditions
    if not (0.9 < norm_a < 100):
        log.error(f"[sanity] FAIL: embedding norm out of expected range")
        return False
    if a_a_cos < 0.999:
        log.error(f"[sanity] FAIL: model is non-deterministic")
        return False
    if a_b_cos > 0.999:
        log.error(f"[sanity] FAIL: model produces same output for different inputs (collapsed?)")
        return False
    log.info(f"[sanity] PASS — encoder is functional and deterministic")
    return True


def slice_segments(audio: np.ndarray, win_samples: int, hop_samples: int) -> np.ndarray:
    n = audio.shape[0]
    if n < win_samples:
        return np.empty((0, 1, win_samples), dtype=np.float32)
    n_seg = (n - win_samples) // hop_samples + 1
    out = np.empty((n_seg, 1, win_samples), dtype=np.float32)
    for i in range(n_seg):
        out[i, 0, :] = audio[i * hop_samples:i * hop_samples + win_samples]
    return out


def encode_audio(m_pre, m_fp, audio: np.ndarray, *,
                 win: int, hop: int, batch: int = 64) -> np.ndarray:
    import tensorflow as tf
    segs = slice_segments(audio, win, hop)
    if segs.shape[0] == 0:
        return np.empty((0, 128), dtype=np.float32)
    out = []
    for s in range(0, segs.shape[0], batch):
        e = min(s + batch, segs.shape[0])
        emb = m_fp(m_pre(tf.constant(segs[s:e]))).numpy()
        out.append(emb)
    emb = np.concatenate(out, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)


def index_refs(m_pre, m_fp, fs: int, win: int, hop: int,
               out_dir: Path, log: logging.Logger):
    """Encode all 357 Saraga refs with NMFP encoder. Save to nmfp_ref_embs.mm."""
    import librosa
    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)

    # First pass: count total segments
    n_segs_per_ref = []
    for _, r in refs.iterrows():
        dur_samples = int(float(r["duration_sec"]) * fs)
        n_segs_per_ref.append(max(0, (dur_samples - win) // hop + 1))
    n_total = sum(n_segs_per_ref)
    log.info(f"[index] estimated total segments: {n_total:,}")

    emb_path = out_dir / "nmfp_ref_embs.mm"
    embs = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total, 128))

    lookup_rows = []
    offset = 0
    t0 = time.time()
    for i, (_, r) in enumerate(refs.iterrows()):
        ref_id = r["ref_id"]
        try:
            audio, _ = librosa.load(r["audio_path"], sr=fs, mono=True)
        except Exception as exc:
            log.error(f"[index] load failed for {ref_id}: {exc}")
            continue
        emb = encode_audio(m_pre, m_fp, audio.astype(np.float32), win=win, hop=hop)
        n_seg = emb.shape[0]
        if n_seg == 0:
            continue
        # Cap to estimate
        n_seg = min(n_seg, n_segs_per_ref[i])
        emb = emb[:n_seg]
        embs[offset:offset + n_seg] = emb
        for si in range(n_seg):
            lookup_rows.append({
                "global_idx": offset + si,
                "ref_id": ref_id,
                "segment_idx": si,
                "segment_start_sec": si * (hop / fs),
            })
        offset += n_seg
        if (i + 1) % 50 == 0 or i == len(refs) - 1:
            log.info(f"[index] {i+1}/{len(refs)}  {ref_id}  segs={n_seg}  "
                     f"elapsed={time.time()-t0:.0f}s  offset={offset:,}")

    # Truncate memmap if over-estimated
    if offset != n_total:
        log.info(f"[index] truncating memmap from {n_total} → {offset}")
        embs.flush()
        del embs
        embs = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(offset, 128))
    embs.flush()

    lookup = pd.DataFrame(lookup_rows)
    lookup.to_parquet(out_dir / "nmfp_ref_segment_lookup.parquet", index=False)
    log.info(f"[index] DONE — {offset:,} segments in {time.time()-t0:.0f}s")
    return embs, lookup, offset


def sequence_search(q_emb: np.ndarray, index, embs: np.memmap, *,
                    k_probe: int = 20, top_k: int = 10):
    L = q_emb.shape[0]
    _, I = index.search(q_emb.astype(np.float32), k_probe)
    for offset in range(L):
        I[offset, :] -= offset
    cands = np.unique(I[I >= 0])
    cands = cands[cands + L <= embs.shape[0]]
    if len(cands) == 0:
        return []
    scores = np.empty(len(cands), dtype=np.float32)
    for ci, cid in enumerate(cands):
        ref_chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, ref_chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="main_1s", help="cell to evaluate")
    ap.add_argument("--queries", nargs="+", default=[
        "data/manifests/hindustani/queries_1s.csv",
        "data/manifests/carnatic/queries_1s.csv",
    ])
    ap.add_argument("--skip-index", action="store_true",
                    help="re-use existing nmfp_ref_embs.mm")
    args = ap.parse_args()

    log = setup_log()
    out_dir = OUT_DIR / f"saraga_{args.cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build encoder + sanity check
    log.info(f"[step 1] building encoder + loading NMFP ckpt-100")
    m_pre, m_fp, cfg = build_and_load_encoder(log)
    fs = int(cfg["MODEL"]["FS"])
    win = int(cfg["MODEL"]["DUR"] * fs)
    hop = int(cfg["MODEL"]["HOP"] * fs)
    log.info(f"[cfg] fs={fs} win={win} hop={hop} F_MIN={cfg['MODEL']['F_MIN']}")

    log.info(f"[step 2] sanity check")
    if not sanity_check_encoder(m_pre, m_fp, log):
        log.error("ABORT — sanity check failed")
        return 1

    # 2) Index refs
    emb_path = OUT_DIR / "nmfp_ref_embs.mm"
    if args.skip_index and emb_path.exists():
        lookup = pd.read_parquet(OUT_DIR / "nmfp_ref_segment_lookup.parquet")
        n = len(lookup)
        embs = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(n, 128))
        log.info(f"[step 3] re-using existing index: {n:,} segments")
    else:
        log.info(f"[step 3] encoding all 357 Saraga refs (will take ~30-60 min on Mac CPU)")
        embs, lookup, n = index_refs(m_pre, m_fp, fs, win, hop, OUT_DIR, log)

    # 3) Build FAISS index
    import faiss
    log.info(f"[step 4] building FAISS IndexFlatIP over {embs.shape[0]:,} segments")
    index = faiss.IndexFlatIP(128)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))

    # 4) Run queries
    log.info(f"[step 5] running queries for cell {args.cell}")
    queries = pd.concat([pd.read_csv(REPO / q) for q in args.queries], ignore_index=True)
    log.info(f"[query] {len(queries)} queries loaded")

    import librosa
    lookup_by_idx = lookup.set_index("global_idx")
    rows = []
    t0 = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        qid = q["query_id"]
        try:
            audio, _ = librosa.load(q["audio_path"], sr=fs, mono=True)
        except Exception:
            rows.append({"query_id": qid, "rank": 0, "predicted_ref_id": None,
                         "nafp_score": None})
            continue
        q_emb = encode_audio(m_pre, m_fp, audio.astype(np.float32),
                             win=win, hop=hop, batch=64)
        if q_emb.shape[0] == 0:
            rows.append({"query_id": qid, "rank": 0, "predicted_ref_id": None,
                         "nafp_score": None})
            continue
        hits = sequence_search(q_emb, index, embs, k_probe=20, top_k=10)
        for rank, (cid, score) in enumerate(hits, start=1):
            info = lookup_by_idx.loc[cid]
            rows.append({
                "query_id": qid, "rank": rank,
                "predicted_ref_id": str(info["ref_id"]),
                "predicted_segment_id": int(cid),
                "nafp_score": float(score),
            })
        if (qi + 1) % 50 == 0:
            log.info(f"[query] {qi+1}/{len(queries)}  elapsed={time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "query_results.parquet", index=False)
    log.info(f"[write] {out_dir/'query_results.parquet'}  ({len(df)} rows)")

    # 5) Compute HR@1 and compare to baseline
    truth = dict(zip(queries.query_id, queries.ref_id))
    top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
    hits = sum(1 for qid in queries.query_id if truth[qid] == top1.get(qid))
    hr1 = hits / len(queries)
    log.info(f"[result] NMFP-ckpt100 HR@1 on {args.cell}: {hr1:.4f}  ({hits}/{len(queries)})")
    log.info(f"[result] NAFP-ckpt10 baseline  HR@1 on {args.cell}: 0.9830  (983/1000)")
    log.info(f"[result] Δ = {hr1 - 0.983:+.4f}")

    scores = {
        "cell": args.cell,
        "n_queries": len(queries),
        "n_hits": hits,
        "hr@1": hr1,
        "baseline_hr@1": 0.983,
        "delta_hr@1": hr1 - 0.983,
        "ckpt": "nmfp-triplet ckpt-100",
        "config": {"F_MIN": cfg["MODEL"]["F_MIN"]},
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
