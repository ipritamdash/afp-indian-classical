"""NAFP system runner — neural audio fingerprinting via TF encoder + FAISS.

Architecture (mirrors the upstream NAFP eval pipeline, adapted to our Saraga manifests):

1. Build the encoder (`build_fp` from upstream `model/generate.py`) — m_pre (log-mel) + m_fp
   (CNN encoder + projection). Load a trained checkpoint from `--checkpoint-dir`.
2. INDEX refs: for each ref audio, resample → 8 kHz mono, slice into 1-sec windows with
   0.5-sec hop, push through encoder in batches → 128-D L2-normalized embeddings. Append
   to a contiguous numpy array + a (ref_id, segment_idx, start_sec) lookup table. Save
   embeddings to `--output/ref_embs.mm` (numpy memmap) so the index step is decoupled from
   query time.
3. BUILD FAISS index: `IndexFlatIP` over the L2-normalized ref embeddings — exact inner
   product == cosine similarity for L2-normalized vectors. CPU-only on Mac (faiss-cpu).
4. QUERY: for each 10-sec query, encode → 19 segments × 128-D. Per-segment top-K search,
   offset-compensate to recover candidate sequence start IDs, rescore via diagonal dot
   product over the full sequence (formula from upstream eval_faiss.py line 224), rank.
5. OUTPUT: `query_results.parquet` matching the contract of the other runners:
   query_id, rank, predicted_ref_id, predicted_segment_id, match_count (=L=19),
   ref_start, ref_stop, query_start, query_stop, nafp_score, latency_ms.

Reproducibility contract:
- Audio loaded via librosa with `mono=True` + `sr=8000` (NAFP's MODEL.FS) — sample-accurate.
- Segment-time offset = segment_idx × HOP (default 0.5 s) per upstream MODEL.HOP.
- L2 normalization on every embedding before indexing (defensive — encoder already emits
  L2-normalized, but cosine-via-IP semantics assume this).
- Single-threaded query path: FAISS IndexFlatIP is already vectorized; no Python parallelism.

Usage:
  uv run python scripts/systems/nafp_runner.py \\
      --refs    data/manifests/hindustani/refs.csv \\
                data/manifests/carnatic/refs.csv \\
      --queries data/manifests/hindustani/queries.csv \\
                data/manifests/carnatic/queries.csv \\
      --checkpoint-dir data/results/nafp/kaggle_output/logs/checkpoint \\
      --checkpoint-name pipeline \\
      --checkpoint-index 10 \\
      --config scripts/nafp/upstream/config/pipeline.yaml \\
      --output  data/results/nafp/saraga_only_main \\
      --top-k   10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# TF compat (must be set BEFORE TF import) — match Kaggle training env exactly so the
# checkpoint loads against the same legacy-Keras stack.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NAFP_UPSTREAM = PROJECT_ROOT / "scripts" / "nafp" / "upstream"
sys.path.insert(0, str(NAFP_UPSTREAM))


# ── manifest helpers ──────────────────────────────────────────────────────

def load_concat_csv(paths: list[str]) -> pd.DataFrame:
    """Concatenate one or more CSV manifests (e.g. hindustani + carnatic)."""
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_config(path: Path) -> dict:
    import yaml
    with path.open() as f:
        return yaml.safe_load(f)


# ── audio → segments ──────────────────────────────────────────────────────

def slice_segments(audio_8k: np.ndarray, win_samples: int, hop_samples: int) -> np.ndarray:
    """Slice a 1-D mono float32 audio array (8 kHz) into windows.
    Returns (n_segments, 1, win_samples) shape ready for m_pre input."""
    n = len(audio_8k)
    if n < win_samples:
        return np.empty((0, 1, win_samples), dtype=np.float32)
    n_seg = (n - win_samples) // hop_samples + 1
    out = np.empty((n_seg, 1, win_samples), dtype=np.float32)
    for i in range(n_seg):
        s = i * hop_samples
        out[i, 0, :] = audio_8k[s:s + win_samples]
    return out


def load_resample_mono(path: str, target_sr: int) -> np.ndarray:
    """Robust audio loader: librosa.load handles MP3, WAV, FLAC etc. via soundfile/audioread."""
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32, copy=False)


# ── encoder build + checkpoint load ───────────────────────────────────────

def build_and_load_encoder(cfg: dict, ckpt_dir: Path, ckpt_name: str,
                           ckpt_index: int | None) -> tuple:
    """Build m_pre + m_fp from upstream NAFP, restore checkpoint. Returns the pair."""
    # imports deferred until after sys.path patched + TF env set
    import tensorflow as tf  # noqa: E402
    from model.generate import build_fp, load_checkpoint  # noqa: E402

    m_pre, m_fp = build_fp(cfg)
    # NAFP's load_checkpoint signature: (checkpoint_root_dir, checkpoint_name, checkpoint_index, m_fp)
    # It internally appends f"/{checkpoint_name}/" to root → pass root WITHOUT name.
    actual_idx = load_checkpoint(str(ckpt_dir), ckpt_name, ckpt_index, m_fp)
    return m_pre, m_fp, actual_idx


def encode_batched(m_pre, m_fp, segments: np.ndarray, batch_size: int) -> np.ndarray:
    """Forward-pass `segments` (n, 1, win) through m_fp(m_pre(·)) in batches.
    Returns (n, EMB_SZ) float32 numpy array.

    Note: tried wrapping in @tf.function on Mac CPU and measured a 33%
    SLOWDOWN (graph-mode overhead exceeds eager per-op overhead for this small
    model + small-ish batches). Eager mode is faster on this hardware."""
    import tensorflow as tf  # noqa: E402
    n = segments.shape[0]
    if n == 0:
        return np.empty((0, 128), dtype=np.float32)
    out: list[np.ndarray] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        x = tf.constant(segments[s:e], dtype=tf.float32)
        emb = m_fp(m_pre(x))           # (b, EMB_SZ); already L2-normalized by NAFP
        out.append(emb.numpy())
    return np.concatenate(out, axis=0)


# ── index refs ────────────────────────────────────────────────────────────

def index_refs(refs_df: pd.DataFrame, m_pre, m_fp, *,
               fs: int, win_samples: int, hop_samples: int, batch_size: int,
               output_dir: Path, log: logging.Logger) -> tuple[np.memmap, pd.DataFrame, dict]:
    """Encode all ref audio into one big embedding memmap.
    Returns (embeddings_memmap, segment_lookup_df, stats_dict)."""

    # First pass: count total segments so we can size the memmap exactly.
    log.info(f"[index] first pass: counting segments across {len(refs_df)} refs")
    n_segments_per_ref: list[int] = []
    durations: list[float] = []
    for i, r in refs_df.iterrows():
        dur = float(r["duration_sec"])
        durations.append(dur)
        # NAFP slicing: floor((duration_samples - win) / hop) + 1
        dur_samples = int(dur * fs)
        n_seg = max(0, (dur_samples - win_samples) // hop_samples + 1)
        n_segments_per_ref.append(n_seg)
    n_total_segments = sum(n_segments_per_ref)
    log.info(f"[index] total segments to encode: {n_total_segments:,}")

    emb_path = output_dir / "ref_embs.mm"
    output_dir.mkdir(parents=True, exist_ok=True)
    embs = np.memmap(emb_path, dtype="float32", mode="w+",
                     shape=(n_total_segments, 128))

    # Second pass: load → encode → write into the memmap at the right offset
    lookup_rows: list[dict] = []
    offset = 0
    t0_total = time.time()
    for i, (_, r) in enumerate(refs_df.iterrows()):
        audio_path = r["audio_path"]
        ref_id = r["ref_id"]
        n_seg_expected = n_segments_per_ref[i]
        if n_seg_expected == 0:
            log.warning(f"[index] skip {ref_id}: too short ({durations[i]:.1f}s)")
            continue
        t0 = time.time()
        try:
            audio = load_resample_mono(audio_path, fs)
        except Exception as exc:
            log.error(f"[index] load failed for {ref_id}: {exc}")
            continue
        segs = slice_segments(audio, win_samples, hop_samples)
        n_seg = segs.shape[0]
        if n_seg == 0:
            log.warning(f"[index] skip {ref_id}: no segments after slice")
            continue
        # Cap to expected count if floating-point duration mismatch caused +/-1 drift
        if n_seg != n_seg_expected:
            log.debug(f"[index] {ref_id}: n_seg={n_seg} expected={n_seg_expected}; trimming")
            n_seg = min(n_seg, n_seg_expected)
            segs = segs[:n_seg]
        emb = encode_batched(m_pre, m_fp, segs, batch_size)
        # L2-renorm defensively
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.clip(norms, 1e-12, None)
        embs[offset:offset + n_seg, :] = emb
        for si in range(n_seg):
            lookup_rows.append({
                "global_idx": offset + si,
                "ref_id": ref_id,
                "segment_idx": si,
                "segment_start_sec": si * (hop_samples / fs),
            })
        offset += n_seg
        log.info(f"[index] {i+1}/{len(refs_df)}  {ref_id}  segs={n_seg}  "
                 f"{time.time()-t0:.2f}s")

    if offset != n_total_segments:
        log.warning(f"[index] actual segments {offset} != predicted {n_total_segments}; truncating memmap")
        embs.flush()
        del embs
        embs = np.memmap(emb_path, dtype="float32", mode="r+",
                         shape=(offset, 128))
    embs.flush()
    lookup = pd.DataFrame(lookup_rows)
    lookup.to_parquet(output_dir / "ref_segment_lookup.parquet", index=False)

    stats = {
        "n_refs": int(len(refs_df)),
        "n_segments_total": int(offset),
        "ref_embs_mb": round(emb_path.stat().st_size / 1e6, 2),
        "index_seconds": round(time.time() - t0_total, 1),
        "fs": fs,
        "win_samples": win_samples,
        "hop_samples": hop_samples,
    }
    return embs, lookup, stats


# ── build FAISS index ─────────────────────────────────────────────────────

def build_faiss_index(embs: np.ndarray):
    """Exact inner-product index over L2-normalized embeddings == cosine similarity."""
    import faiss
    n, d = embs.shape
    index = faiss.IndexFlatIP(d)
    # Convert memmap to contiguous float32 array for index.add (faiss requires contiguous)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))
    return index


# ── sequence-level search ─────────────────────────────────────────────────

def sequence_search(query_emb: np.ndarray, index, embs: np.memmap, *,
                    k_probe: int, top_k: int) -> list[tuple[int, float]]:
    """Per upstream eval_faiss.py:
       - Per-segment top-K search → matrix I of shape (L, k_probe)
       - Offset-compensate: I[offset, :] -= offset → recovers sequence start IDs
       - Unique candidates ≥ 0
       - For each candidate cid: score = mean(diag(query_emb @ embs[cid:cid+L].T))
       - Return top-`top_k` (candidate_global_idx, score) sorted desc by score."""
    L = query_emb.shape[0]
    _, I = index.search(query_emb.astype(np.float32, copy=False), k_probe)
    for offset in range(L):
        I[offset, :] -= offset
    candidates = np.unique(I[I >= 0])
    n_total = embs.shape[0]
    # Drop candidates that would run off the end of the index
    candidates = candidates[candidates + L <= n_total]
    if len(candidates) == 0:
        return []
    scores = np.empty(len(candidates), dtype=np.float32)
    for ci, cid in enumerate(candidates):
        ref_chunk = np.asarray(embs[cid:cid + L])
        # mean of diagonal of (query @ ref.T) == mean over segments of cosine sim
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", query_emb, ref_chunk)))
    # kind='stable' for deterministic tie-breaking (no impact on current data
    # which has no exact ties, but future-proofs reproducibility)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(candidates[o]), float(scores[o])) for o in order]


# ── query loop ────────────────────────────────────────────────────────────

def run_queries(queries_df: pd.DataFrame, lookup: pd.DataFrame, embs: np.memmap, index,
                m_pre, m_fp, *, fs: int, win_samples: int, hop_samples: int,
                batch_size: int, k_probe: int, top_k: int,
                log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """For each query: encode → sequence search → emit top-K rows."""
    rows: list[dict] = []
    latencies_ms: list[float] = []
    n_no_match = 0
    lookup_by_idx = lookup.set_index("global_idx")[["ref_id", "segment_start_sec"]]

    for qi, (_, q) in enumerate(queries_df.iterrows()):
        query_id = q["query_id"]
        audio_path = q["audio_path"]
        try:
            audio = load_resample_mono(audio_path, fs)
        except Exception as exc:
            log.error(f"[query] load failed for {query_id}: {exc}")
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": None})
            continue
        segs = slice_segments(audio, win_samples, hop_samples)
        if segs.shape[0] == 0:
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": None})
            continue
        t0 = time.time()
        q_emb = encode_batched(m_pre, m_fp, segs, batch_size)
        # defensive L2-renorm
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb = q_emb / np.clip(norms, 1e-12, None)

        hits = sequence_search(q_emb, index, embs, k_probe=k_probe, top_k=top_k)
        latency_ms = (time.time() - t0) * 1000
        latencies_ms.append(latency_ms)

        if not hits:
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": latency_ms})
            continue

        L = q_emb.shape[0]
        seg_dur_sec = win_samples / fs  # 1.0 s
        seg_hop_sec = hop_samples / fs  # 0.5 s
        seq_dur_sec = (L - 1) * seg_hop_sec + seg_dur_sec  # 10.0 s for L=19
        for rank, (cid, score) in enumerate(hits, start=1):
            ref_id = lookup_by_idx.at[cid, "ref_id"]
            ref_start = float(lookup_by_idx.at[cid, "segment_start_sec"])
            rows.append({
                "query_id": query_id,
                "rank": rank,
                "predicted_ref_id": ref_id,
                "predicted_segment_id": cid,
                "match_count": L,         # sequence length used (NAFP analogue of "matches")
                "ref_start": ref_start,
                "ref_stop": ref_start + seq_dur_sec,
                "query_start": 0.0,
                "query_stop": seq_dur_sec,
                "nafp_score": score,
                "latency_ms": latency_ms,
            })
        if (qi + 1) % 50 == 0:
            log.info(f"[query] {qi+1}/{len(queries_df)}  last={query_id}  "
                     f"top1_score={hits[0][1]:.4f}  latency={latency_ms:.0f}ms")

    stats = {
        "n_queries": int(len(queries_df)),
        "n_no_match": int(n_no_match),
        "frac_no_match": round(n_no_match / max(1, len(queries_df)), 4),
        "latency_ms_p50": float(np.median(latencies_ms)) if latencies_ms else None,
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
        "latency_ms_p99": float(np.percentile(latencies_ms, 99)) if latencies_ms else None,
        "k_probe": k_probe,
        "top_k": top_k,
    }
    return pd.DataFrame(rows), stats


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, help="ref manifest CSV(s)")
    ap.add_argument("--queries", nargs="+", required=True, help="query manifest CSV(s)")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="dir CONTAINING the checkpoint-name subdir (NAFP's load_checkpoint joins these)")
    ap.add_argument("--checkpoint-name", default="pipeline")
    ap.add_argument("--checkpoint-index", type=int, default=None,
                    help="ckpt-N index. Omit to load latest.")
    ap.add_argument("--config", default=str(NAFP_UPSTREAM / "config" / "default.yaml"),
                    help="NAFP pipeline.yaml (or default.yaml). Must match training config.")
    ap.add_argument("--output", required=True, help="output directory for parquet + JSON")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--k-probe", type=int, default=20,
                    help="per-segment top-K for FAISS search. Default 20 per NAFP paper.")
    ap.add_argument("--batch-size", type=int, default=125, help="encoder forward batch size")
    ap.add_argument("--skip-index", action="store_true",
                    help="skip indexing; reuse ref_embs.mm + ref_segment_lookup.parquet + index_log.json "
                         "from --index-from (or --output if --index-from is omitted). "
                         "Mirrors the --skip-index convention of olaf/dejavu/panako runners.")
    ap.add_argument("--index-from", type=Path, default=None,
                    help="directory to read existing ref index from when --skip-index is set. "
                         "Defaults to --output. Must contain ref_embs.mm, ref_segment_lookup.parquet, "
                         "and index_log.json.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("nafp_runner")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load config + manifests
    cfg = load_config(Path(args.config))
    fs = int(cfg["MODEL"]["FS"])                       # 8000
    win_samples = int(fs * float(cfg["MODEL"]["DUR"])) # 8000
    hop_samples = int(fs * float(cfg["MODEL"]["HOP"])) # 4000
    log.info(f"cfg: fs={fs} win={win_samples} hop={hop_samples} EMB_SZ={cfg['MODEL']['EMB_SZ']}")

    refs_df = load_concat_csv(args.refs)
    queries_df = load_concat_csv(args.queries)
    log.info(f"manifests: {len(refs_df)} refs, {len(queries_df)} queries")

    # 2) Build encoder + load checkpoint
    log.info("building encoder + loading checkpoint")
    t0 = time.time()
    m_pre, m_fp, actual_ckpt_idx = build_and_load_encoder(
        cfg, Path(args.checkpoint_dir), args.checkpoint_name, args.checkpoint_index,
    )
    log.info(f"encoder ready in {time.time()-t0:.1f}s (ckpt-{actual_ckpt_idx})")

    # 3) Index refs — either re-encode from scratch, or reuse a previous index.
    if args.skip_index:
        idx_dir = Path(args.index_from) if args.index_from else out_dir
        log.info(f"[skip-index] reusing ref index from {idx_dir}")
        idx_log_path = idx_dir / "index_log.json"
        emb_path     = idx_dir / "ref_embs.mm"
        lookup_path  = idx_dir / "ref_segment_lookup.parquet"
        for p in (idx_log_path, emb_path, lookup_path):
            if not p.exists():
                log.error(f"--skip-index: missing artefact at {p}")
                return 2
        idx_stats = json.loads(idx_log_path.read_text())
        n_seg = int(idx_stats["n_segments_total"])
        emb_dim = int(cfg["MODEL"]["EMB_SZ"])
        embs = np.memmap(emb_path, dtype="float32", mode="r", shape=(n_seg, emb_dim))
        lookup = pd.read_parquet(lookup_path)
        # Sanity: the lookup parquet must have exactly n_seg rows
        assert len(lookup) == n_seg, \
            f"lookup parquet has {len(lookup)} rows, expected {n_seg} (from index_log.json)"
        # Sanity: re-checking the ref-set used to build the index matches the one we're about to query
        if int(idx_stats.get("n_refs", -1)) != len(refs_df):
            log.warning(f"refs count mismatch: index_log says {idx_stats.get('n_refs')}, "
                        f"refs manifest has {len(refs_df)}. Proceeding anyway — only the index is reused.")
        log.info(f"[skip-index] reusing {n_seg:,} segments × {emb_dim}-D from {emb_path}")
    else:
        embs, lookup, idx_stats = index_refs(
            refs_df, m_pre, m_fp,
            fs=fs, win_samples=win_samples, hop_samples=hop_samples,
            batch_size=args.batch_size, output_dir=out_dir, log=log,
        )
        idx_stats["checkpoint_index"] = int(actual_ckpt_idx)
        (out_dir / "index_log.json").write_text(json.dumps(idx_stats, indent=2))
        log.info(f"index done: {idx_stats}")

    # 4) Build FAISS index (exact IP over L2-normalized vectors)
    log.info("building FAISS IndexFlatIP")
    t0 = time.time()
    index = build_faiss_index(np.asarray(embs))
    log.info(f"faiss index built in {time.time()-t0:.1f}s  ntotal={index.ntotal}")

    # 5) Run queries
    results_df, q_stats = run_queries(
        queries_df, lookup, embs, index, m_pre, m_fp,
        fs=fs, win_samples=win_samples, hop_samples=hop_samples,
        batch_size=args.batch_size, k_probe=args.k_probe, top_k=args.top_k,
        log=log,
    )

    # 6) Write outputs
    results_df.to_parquet(out_dir / "query_results.parquet", index=False)
    (out_dir / "query_log.json").write_text(json.dumps(q_stats, indent=2))
    log.info(f"done: {q_stats}")
    log.info(f"results → {out_dir}/query_results.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
