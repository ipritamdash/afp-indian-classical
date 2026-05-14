"""Intervention 2 — α-sweep + controls scorer.

For ONE cell (NAFP × Saraga × length), this script:
  1. Loads ref_embs + index + centroids + queries + encoder
  2. Encodes each query ONCE (eager TF, slow part)
  3. Runs the FAISS search ONCE per query at k_probe=100 (wider than baseline 20)
  4. For each α ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
     AND for each control ∈ {shuffled-centroid, isotropic-centroid}
        re-scores the shared candidate pool and emits top-10 parquet rows

Output (per cell, into <out_dir>):
  query_results_alpha_<a>.parquet        (7 files)
  query_results_control_<name>.parquet   (2 files)
  scores_alpha_<a>.json                  (computed by score.py separately)
  q_encode_meta.json                     (timing + per-query candidate counts)

α=0 is the bit-identical sanity baseline (zero subtraction → renorm doesn't move
already-L2-normalized embeddings → ranking identical to k_probe=100 baseline).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── env: silence TF + force CPU eager + LEGACY KERAS (mandatory) ─────────
# TF_USE_LEGACY_KERAS=1 is REQUIRED. Without it, tf.train.Checkpoint silently
# partial-restores ckpt-10 due to Keras-3 vs tf-keras variable-naming drift,
# leaving ~64 CNN/LayerNorm vars at random init → embeddings useless.
# Verified 2026-05-13: setting this gives 0 unmatched vars + cos≥0.98 vs
# Kaggle-encoded refs (residual gap is Apple-Silicon vs CUDA float precision).
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

REPO = Path(__file__).resolve().parents[3]
NAFP_DIR = REPO / "data/results/nafp/saraga_only_main"
CENT_DIR = REPO / "data/results/nafp/intervention2/centroids"
OUT_ROOT = REPO / "data/results/nafp/intervention2"

ALPHA_SWEEP = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CONTROL_ALPHA = 0.10   # controls evaluated at the pre-registered headline α
SEED_SHUFFLE = 20260512
SEED_ISO = 20260513


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(__name__)


def build_control_centroids(centroids: np.ndarray, artist_per_ref: np.ndarray,
                            log: logging.Logger) -> dict[str, np.ndarray]:
    """Construct shuffled-centroid + isotropic-centroid controls."""
    rng_s = np.random.default_rng(SEED_SHUFFLE)
    rng_i = np.random.default_rng(SEED_ISO)

    # ── shuffled: each ref gets the centroid of a DIFFERENT artist's ref ─
    n_refs = centroids.shape[0]
    shuffled = np.zeros_like(centroids)
    donor_artists = []
    for r in range(n_refs):
        cand_idx = np.where(artist_per_ref != artist_per_ref[r])[0]
        donor = int(rng_s.choice(cand_idx))
        shuffled[r] = centroids[donor]
        donor_artists.append(artist_per_ref[donor])
    n_diff = sum(1 for r in range(n_refs)
                 if donor_artists[r] != artist_per_ref[r])
    log.info(f"[control] shuffled-centroid built, seed={SEED_SHUFFLE} "
             f"({n_diff}/{n_refs} refs paired with different-artist centroid)")
    assert n_diff == n_refs, "shuffled control must guarantee different-artist donor"

    # ── isotropic: random unit-Gaussian × original norm ──────────────────
    iso = rng_i.normal(size=centroids.shape).astype(np.float32)
    iso = iso / np.linalg.norm(iso, axis=1, keepdims=True)
    iso = iso * np.linalg.norm(centroids, axis=1, keepdims=True)
    log.info(f"[control] isotropic-centroid built, seed={SEED_ISO} "
             f"(unit-Gaussian × matching norms)")

    return {"shuffled": shuffled, "isotropic": iso}


def setup_encoder(repo: Path, log: logging.Logger):
    """Build NAFP encoder + load ckpt-10 (the production checkpoint)."""
    upstream = repo / "scripts/nafp/upstream"
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    cfg_path = repo / "scripts/nafp/upstream/config/default.yaml"
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = repo / "data/results/nafp/kaggle_output/logs/checkpoint"
    log.info(f"[encoder] building from cfg {cfg_path}")

    import tensorflow as tf
    tf.keras.backend.clear_session()  # reset Keras naming counters (mandatory)
    from model.generate import build_fp
    m_pre, m_fp = build_fp(cfg)
    _ = m_fp(m_pre(tf.zeros((1, 1, 8000), dtype=tf.float32))).numpy()  # build sublayers

    # Manual restore. Checkpoint contains optimizer state we don't need (~1100 vars);
    # we use expect_partial() to ignore those, then verify MODEL vars are restored
    # by comparing one Mac-encoded ref segment to the stored Kaggle-encoded version.
    checkpoint = tf.train.Checkpoint(model=m_fp)
    status = checkpoint.restore(str(ckpt_dir / "pipeline" / "ckpt-10"))
    status.expect_partial()  # silences optimizer var warnings
    # Don't call assert_existing_objects_matched yet — it may flag legitimate
    # framework-managed extras. Validate empirically via the sanity check below.
    log.info(f"[encoder] loaded ckpt-10 (model vars restored; optimizer state skipped)")
    return m_pre, m_fp, cfg


def load_resample_mono(path: str, target_sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32, copy=False)


def slice_segments(audio: np.ndarray, win_samples: int, hop_samples: int) -> np.ndarray:
    n = audio.shape[0]
    if n < win_samples:
        return np.empty((0, 1, win_samples), dtype=np.float32)
    n_seg = (n - win_samples) // hop_samples + 1
    out = np.empty((n_seg, 1, win_samples), dtype=np.float32)
    for i in range(n_seg):
        s = i * hop_samples
        out[i, 0, :] = audio[s:s + win_samples]
    return out


def encode_query(m_pre, m_fp, audio_path: str, *,
                 fs: int, win_samples: int, hop_samples: int) -> np.ndarray | None:
    import tensorflow as tf
    try:
        audio = load_resample_mono(audio_path, fs)
    except Exception as exc:
        return None
    segs = slice_segments(audio, win_samples, hop_samples)
    if segs.shape[0] == 0:
        return None
    x = tf.constant(segs, dtype=tf.float32)
    emb = m_fp(m_pre(x)).numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)


def rescore_candidates(q_emb: np.ndarray, candidates: np.ndarray,
                       embs: np.memmap, seg_to_ref: np.ndarray,
                       mu: np.ndarray | None, alpha: float,
                       top_k: int) -> list[tuple[int, float]]:
    """Re-rank `candidates` under per-segment artist-centroid subtraction.

    mu : (357, 128) centroid array (or None for pure baseline).
    alpha=0 ⇒ baseline ranking (must be identical regardless of mu).
    """
    L = q_emb.shape[0]
    scores = np.empty(len(candidates), dtype=np.float32)
    for ci, cid in enumerate(candidates):
        ref_chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        if alpha != 0.0 and mu is not None:
            seg_refs = seg_to_ref[cid:cid + L]            # (L,)
            mu_chunk = mu[seg_refs]                         # (L, 128)
            ref_chunk = ref_chunk - alpha * mu_chunk
            norms = np.linalg.norm(ref_chunk, axis=1, keepdims=True)
            ref_chunk = ref_chunk / np.clip(norms, 1e-12, None)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, ref_chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(candidates[o]), float(scores[o])) for o in order]


def emit_rows(query_id: str, hits: list[tuple[int, float]], lookup_by_idx: pd.DataFrame,
              q_length_sec: float, L: int, hop_sec: float,
              latency_ms: float) -> list[dict]:
    rows = []
    if not hits:
        rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                     "predicted_segment_id": None, "match_count": 0,
                     "ref_start": None, "ref_stop": None,
                     "query_start": None, "query_stop": None,
                     "nafp_score": None, "latency_ms": latency_ms})
        return rows
    for rank, (cid, score) in enumerate(hits, start=1):
        info = lookup_by_idx.loc[cid]
        rows.append({
            "query_id": query_id,
            "rank": rank,
            "predicted_ref_id": str(info["ref_id"]),
            "predicted_segment_id": int(cid),
            "match_count": L,
            "ref_start": float(info["segment_start_sec"]),
            "ref_stop": float(info["segment_start_sec"]) + q_length_sec,
            "query_start": 0.0,
            "query_stop": q_length_sec,
            "nafp_score": float(score),
            "latency_ms": latency_ms,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", nargs="+", required=True,
                    help="query manifest CSV(s) — both H + C")
    ap.add_argument("--cell-name", required=True,
                    help="e.g. main_1s, main_10s, ablation_1s")
    ap.add_argument("--k-probe", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N queries (for dev/triage)")
    ap.add_argument("--out", default=None, help="output dir (default OUT_ROOT/<cell-name>)")
    args = ap.parse_args()

    log = setup_logging()
    out_dir = Path(args.out) if args.out else (OUT_ROOT / args.cell_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load shared artifacts ────────────────────────────────────────────
    log.info(f"[load] centroids + segment_to_ref + lookup")
    centroids = np.load(CENT_DIR / "centroids_per_ref.npy")
    seg_to_ref = np.load(CENT_DIR / "segment_to_ref_idx.npy")
    artist_table = pd.read_parquet(CENT_DIR / "artist_table.parquet")
    artist_per_ref = artist_table["artist_key"].values
    lookup = pd.read_parquet(NAFP_DIR / "ref_segment_lookup.parquet").set_index("global_idx")
    embs = np.memmap(NAFP_DIR / "ref_embs.mm", dtype=np.float32, mode="r",
                     shape=(690414, 128))

    log.info(f"[load] queries")
    qdfs = [pd.read_csv(q) for q in args.queries]
    queries = pd.concat(qdfs, ignore_index=True)
    if args.limit is not None:
        queries = queries.head(args.limit)
    log.info(f"[load] {len(queries)} queries")

    # ── controls ─────────────────────────────────────────────────────────
    log.info(f"[control] building shuffled + isotropic centroids")
    controls = build_control_centroids(centroids, artist_per_ref, log)

    # ── faiss index over baseline embs ───────────────────────────────────
    log.info(f"[faiss] building IndexFlatIP over {embs.shape[0]:,} segments")
    import faiss
    t0 = time.time()
    index = faiss.IndexFlatIP(128)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))
    log.info(f"[faiss] index built in {time.time()-t0:.1f}s")

    # ── encoder ──────────────────────────────────────────────────────────
    m_pre, m_fp, cfg = setup_encoder(REPO, log)
    fs = int(cfg["MODEL"]["FS"])
    win_samples = int(cfg["MODEL"]["DUR"] * fs)
    hop_samples = int(cfg["MODEL"]["HOP"] * fs)
    hop_sec = float(cfg["MODEL"]["HOP"])
    log.info(f"[encoder] fs={fs} win={win_samples} hop={hop_samples}")

    # ── encoder reproducibility sanity (FAIL FAST if Mac diverges from Kaggle) ──
    # Encode a known ref segment locally; compare cos vs stored Kaggle embedding.
    # Empirically verified 2026-05-13: cos≈0.98 with TF_USE_LEGACY_KERAS=1.
    import tensorflow as tf
    import librosa
    refs_for_sanity = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)
    sanity_ref = refs_for_sanity[refs_for_sanity.ref_id == "hindustani_0_Raag_Shree"].iloc[0]
    a, _ = librosa.load(sanity_ref["audio_path"], sr=fs, mono=True)
    sanity_seg = a[1028 * hop_samples:1028 * hop_samples + win_samples].reshape(1, 1, win_samples).astype(np.float32)
    sanity_emb = m_fp(m_pre(tf.constant(sanity_seg))).numpy()
    sanity_emb = sanity_emb / np.linalg.norm(sanity_emb)
    stored = embs[1028]
    cos_sanity = float(sanity_emb[0] @ stored)
    log.info(f"[sanity] encoder reproducibility cos(mac, kaggle_stored) = {cos_sanity:.4f}")
    assert cos_sanity > 0.95, (
        f"Mac encoder diverged from Kaggle reference embeddings (cos={cos_sanity:.4f} < 0.95). "
        f"ABORT to avoid invalid intervention results. Likely cause: "
        f"TF_USE_LEGACY_KERAS env var not set BEFORE TF import."
    )

    # ── per-query loop ───────────────────────────────────────────────────
    all_rows: dict[str, list[dict]] = {}
    for a in ALPHA_SWEEP:
        all_rows[f"alpha_{a:.2f}"] = []
    all_rows["control_shuffled"] = []
    all_rows["control_isotropic"] = []

    encode_times: list[float] = []
    rescore_times: list[float] = []
    cand_counts: list[int] = []
    n_no_match = 0

    t_global = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        qid = q["query_id"]
        audio_path = q["audio_path"]
        q_length_sec = float(q["length_sec"])

        t_enc = time.time()
        q_emb = encode_query(m_pre, m_fp, audio_path, fs=fs,
                             win_samples=win_samples, hop_samples=hop_samples)
        encode_times.append(time.time() - t_enc)

        if q_emb is None or q_emb.shape[0] == 0:
            n_no_match += 1
            for k in all_rows:
                all_rows[k].extend(emit_rows(qid, [], lookup, q_length_sec, 0, hop_sec, None))
            continue

        L = q_emb.shape[0]
        # ── FAISS once ───────────────────────────────────────────────────
        _, I = index.search(q_emb.astype(np.float32, copy=False), args.k_probe)
        for offset in range(L):
            I[offset, :] -= offset
        candidates = np.unique(I[I >= 0])
        candidates = candidates[candidates + L <= embs.shape[0]]
        cand_counts.append(len(candidates))

        if len(candidates) == 0:
            n_no_match += 1
            for k in all_rows:
                all_rows[k].extend(emit_rows(qid, [], lookup, q_length_sec, L, hop_sec, None))
            continue

        # ── α sweep ──────────────────────────────────────────────────────
        t_re = time.time()
        for a in ALPHA_SWEEP:
            hits = rescore_candidates(q_emb, candidates, embs, seg_to_ref,
                                      centroids, alpha=a, top_k=args.top_k)
            all_rows[f"alpha_{a:.2f}"].extend(
                emit_rows(qid, hits, lookup, q_length_sec, L, hop_sec, None)
            )
        # ── controls ─────────────────────────────────────────────────────
        for cname, cmu in controls.items():
            hits = rescore_candidates(q_emb, candidates, embs, seg_to_ref,
                                      cmu, alpha=CONTROL_ALPHA, top_k=args.top_k)
            all_rows[f"control_{cname}"].extend(
                emit_rows(qid, hits, lookup, q_length_sec, L, hop_sec, None)
            )
        rescore_times.append(time.time() - t_re)

        if (qi + 1) % 20 == 0 or qi == len(queries) - 1:
            log.info(f"[q] {qi+1}/{len(queries)}  "
                     f"L={L} cands={len(candidates)}  "
                     f"enc={encode_times[-1]:.2f}s  re={rescore_times[-1]:.2f}s  "
                     f"elapsed={time.time()-t_global:.0f}s")

    # ── write parquets ───────────────────────────────────────────────────
    for k, rows in all_rows.items():
        df = pd.DataFrame(rows)
        out_path = out_dir / f"query_results_{k}.parquet"
        df.to_parquet(out_path, index=False)
        log.info(f"[write] {out_path.name}  ({len(df)} rows)")

    # ── meta ────────────────────────────────────────────────────────────
    meta = {
        "cell_name": args.cell_name,
        "n_queries": int(len(queries)),
        "n_no_match": int(n_no_match),
        "k_probe": int(args.k_probe),
        "top_k": int(args.top_k),
        "alpha_sweep": ALPHA_SWEEP,
        "controls": list(controls.keys()),
        "control_alpha": CONTROL_ALPHA,
        "seed_shuffle": SEED_SHUFFLE,
        "seed_isotropic": SEED_ISO,
        "encode_seconds_median": float(np.median(encode_times)) if encode_times else None,
        "rescore_seconds_median": float(np.median(rescore_times)) if rescore_times else None,
        "candidates_median": int(np.median(cand_counts)) if cand_counts else 0,
        "candidates_p95": int(np.percentile(cand_counts, 95)) if cand_counts else 0,
        "total_seconds": round(time.time() - t_global, 1),
    }
    with open(out_dir / "sweep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[done] {meta}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
