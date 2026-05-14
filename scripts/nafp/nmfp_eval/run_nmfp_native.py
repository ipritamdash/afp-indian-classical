"""Move 1 (revised): Run NMFP-published weights on Saraga using NMFP's NATIVE pipeline.

Loads:
  - NMFP nnfp.py FingerPrinter (different from upstream NAFP nnfp.py — different
    norm dim, structure for triplet+past-context training)
  - NMFP essentia mel-spec (T=33 frames for 1s @ 8kHz, vs NAFP kapre T=32)
  - ckpt-100 weights

GATE: does NMFP-as-published beat NAFP-ckpt-10 (HR@1=0.983) on Saraga clean 1s queries?
"""
from __future__ import annotations
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
NMFP_REPO = Path("/tmp/neural-music-fp")
NMFP_CKPT = REPO / "data/nmfp/nmfp-triplet"
OUT_DIR = REPO / "data/results/nafp/nmfp_eval"


def setup_log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(__name__)


def build_encoder(log):
    """Use NMFP's own code to build the FingerPrinter + load ckpt-100."""
    if str(NMFP_REPO) not in sys.path:
        sys.path.insert(0, str(NMFP_REPO))
    import tensorflow as tf
    tf.keras.backend.clear_session()
    from nmfp.model.utils import get_fingerprinter, get_checkpoint_index_and_restore_model
    from nmfp.utils import load_config

    cfg = load_config(str(NMFP_CKPT / "config.yaml"))
    log.info(f"[cfg] F_MIN={cfg['MODEL']['INPUT']['F_MIN']}  N_MELS={cfg['MODEL']['INPUT']['N_MELS']}")
    m_fp = get_fingerprinter(cfg, trainable=False)
    actual_idx = get_checkpoint_index_and_restore_model(m_fp, str(NMFP_CKPT), 100)
    log.info(f"[ckpt] Restored from ckpt-{actual_idx}")
    return m_fp, cfg


def build_melspec(cfg, log):
    """Build NMFP's essentia mel-spec front-end."""
    from nmfp.audio_processing.melspectrogram import Melspec_layer
    inp = cfg["MODEL"]["INPUT"]
    mel = Melspec_layer(
        n_fft=inp["STFT_WIN"],
        stft_hop=inp["STFT_HOP"],
        n_mels=inp["N_MELS"],
        fs=cfg["MODEL"]["AUDIO"]["FS"],
        segment_duration=cfg["MODEL"]["AUDIO"]["SEGMENT_DUR"],
        f_min=inp["F_MIN"],
        f_max=inp["F_MAX"],
        scale=inp.get("SCALE", True),
        dynamic_range=inp.get("DYNAMIC_RANGE", 80),
    )
    log.info(f"[mel] essentia mel-spec ready: {inp['N_MELS']} mels × T(8000@hop{inp['STFT_HOP']})")
    return mel


def slice_audio(audio: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Slice 1-D audio into (n, win) overlapping segments. Same convention as NAFP."""
    if audio.shape[0] < win:
        return np.empty((0, win), dtype=np.float32)
    n = (audio.shape[0] - win) // hop + 1
    out = np.empty((n, win), dtype=np.float32)
    for i in range(n):
        out[i] = audio[i * hop:i * hop + win]
    return out


def encode_audio(audio: np.ndarray, m_fp, mel, *,
                 fs: int, win: int, hop: int, batch: int = 128) -> np.ndarray:
    """Slice → mel-spec → encoder → L2-normalize. Returns (n_seg, 128).
    Streams through audio in batches to bound RAM (Mac CPU std::bad_alloc otherwise)."""
    import tensorflow as tf
    segs = slice_audio(audio.astype(np.float32), win, hop)
    n = segs.shape[0]
    if n == 0:
        return np.empty((0, 128), dtype=np.float32)
    out = np.empty((n, 128), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        # essentia is serial per-segment
        mels = np.stack([mel.compute(segs[i]) for i in range(s, e)], axis=0)  # (b, F, T)
        x = mels[:, :, :, None].astype(np.float32)  # NMFP: (B, F, T, 1)
        emb = m_fp(tf.constant(x), training=False).numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        out[s:e] = emb / np.clip(norms, 1e-12, None)
    return out


def sanity_check(m_fp, mel, log, fs, win, hop) -> bool:
    """Encode 2 distinct Saraga segments + same segment twice; verify model is functional."""
    import librosa
    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)
    ref = refs[refs.ref_id == "hindustani_0_Raag_Shree"].iloc[0]
    audio, _ = librosa.load(ref["audio_path"], sr=fs, mono=True)
    a = audio[1028 * 4000:1028 * 4000 + 8000].astype(np.float32)
    b = audio[2000 * 4000:2000 * 4000 + 8000].astype(np.float32)

    emb_a1 = encode_audio(a, m_fp, mel, fs=fs, win=win, hop=hop)
    emb_a2 = encode_audio(a, m_fp, mel, fs=fs, win=win, hop=hop)
    emb_b = encode_audio(b, m_fp, mel, fs=fs, win=win, hop=hop)

    cos_aa = float(emb_a1[0] @ emb_a2[0])
    cos_ab = float(emb_a1[0] @ emb_b[0])
    log.info(f"[sanity] norm(emb)≈{np.linalg.norm(emb_a1[0]):.4f}")
    log.info(f"[sanity] determinism cos(a,a) = {cos_aa:.6f}  (must ≥0.999)")
    log.info(f"[sanity] discriminability cos(a,b) = {cos_ab:.4f}  (must <0.999)")

    if cos_aa < 0.999:
        log.error("[sanity] FAIL: model non-deterministic")
        return False
    if cos_ab > 0.999:
        log.error("[sanity] FAIL: model collapsed")
        return False
    log.info("[sanity] PASS")
    return True


def index_refs(m_fp, mel, out_dir: Path, log, fs: int, win: int, hop: int):
    import librosa
    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)

    # First pass: estimate total segments
    n_per_ref = [max(0, (int(float(r["duration_sec"]) * fs) - win) // hop + 1)
                 for _, r in refs.iterrows()]
    n_total_est = sum(n_per_ref)
    log.info(f"[index] estimated {n_total_est:,} segments across {len(refs)} refs")

    emb_path = out_dir / "nmfp_ref_embs.mm"
    embs = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total_est, 128))
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
        emb = encode_audio(audio, m_fp, mel, fs=fs, win=win, hop=hop)
        n = min(emb.shape[0], n_per_ref[i])
        if n == 0:
            continue
        embs[offset:offset + n] = emb[:n]
        for si in range(n):
            lookup_rows.append({
                "global_idx": offset + si,
                "ref_id": ref_id,
                "segment_idx": si,
                "segment_start_sec": si * (hop / fs),
            })
        offset += n
        if (i + 1) % 25 == 0 or i == len(refs) - 1:
            log.info(f"[index] {i+1}/{len(refs)}  {ref_id}  +{n} segs  total={offset:,}  "
                     f"elapsed={time.time()-t0:.0f}s")

    if offset != n_total_est:
        log.info(f"[index] truncating memmap {n_total_est} → {offset}")
        embs.flush(); del embs
        embs = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(offset, 128))
    embs.flush()
    lookup = pd.DataFrame(lookup_rows)
    lookup.to_parquet(out_dir / "nmfp_ref_segment_lookup.parquet", index=False)
    log.info(f"[index] DONE — {offset:,} segments in {time.time()-t0:.0f}s")
    return embs, lookup, offset


def sequence_search(q_emb, index, embs, *, k_probe=20, top_k=10):
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
        ref_chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, ref_chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="main_1s")
    ap.add_argument("--queries", nargs="+", default=[
        "data/manifests/hindustani/queries_1s.csv",
        "data/manifests/carnatic/queries_1s.csv"])
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    log = setup_log()
    out_dir = OUT_DIR / f"saraga_{args.cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[step 1] build encoder + load NMFP ckpt-100")
    m_fp, cfg = build_encoder(log)
    fs = int(cfg["MODEL"]["AUDIO"]["FS"])
    win = int(cfg["MODEL"]["AUDIO"]["SEGMENT_DUR"] * fs)
    hop = win // 2

    log.info("[step 2] build essentia mel-spec front-end")
    mel = build_melspec(cfg, log)

    log.info("[step 3] sanity check")
    if not sanity_check(m_fp, mel, log, fs, win, hop):
        return 1

    log.info("[step 4] index refs")
    emb_path = OUT_DIR / "nmfp_ref_embs.mm"
    lookup_path = OUT_DIR / "nmfp_ref_segment_lookup.parquet"
    if args.skip_index and emb_path.exists():
        lookup = pd.read_parquet(lookup_path)
        n = len(lookup)
        embs = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(n, 128))
        log.info(f"[index] re-use existing — {n:,} segments")
    else:
        embs, lookup, n = index_refs(m_fp, mel, OUT_DIR, log, fs, win, hop)

    log.info("[step 5] FAISS index")
    import faiss
    index = faiss.IndexFlatIP(128)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))

    log.info(f"[step 6] queries — cell={args.cell}")
    import librosa
    queries = pd.concat([pd.read_csv(REPO / q) for q in args.queries], ignore_index=True)
    truth = dict(zip(queries.query_id, queries.ref_id))
    lookup_idx = lookup.set_index("global_idx")
    rows = []
    t0 = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        qid = q["query_id"]
        try:
            audio, _ = librosa.load(q["audio_path"], sr=fs, mono=True)
        except Exception:
            rows.append({"query_id": qid, "rank": 0, "predicted_ref_id": None, "nafp_score": None})
            continue
        q_emb = encode_audio(audio, m_fp, mel, fs=fs, win=win, hop=hop)
        if q_emb.shape[0] == 0:
            rows.append({"query_id": qid, "rank": 0, "predicted_ref_id": None, "nafp_score": None})
            continue
        hits = sequence_search(q_emb, index, embs, k_probe=20, top_k=10)
        for rank, (cid, score) in enumerate(hits, start=1):
            info = lookup_idx.loc[cid]
            rows.append({
                "query_id": qid, "rank": rank,
                "predicted_ref_id": str(info["ref_id"]),
                "predicted_segment_id": int(cid),
                "nafp_score": float(score),
            })
        if (qi + 1) % 50 == 0:
            log.info(f"[q] {qi+1}/{len(queries)} elapsed={time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "query_results.parquet", index=False)

    top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
    hits = sum(1 for qid in queries.query_id if truth[qid] == top1.get(qid))
    hr1 = hits / len(queries)
    log.info(f"\n{'='*60}\nNMFP-ckpt100 HR@1 on {args.cell}: {hr1:.4f}  ({hits}/{len(queries)})")
    log.info(f"NAFP-ckpt10 baseline:               0.9830  (983/1000)")
    log.info(f"Δ HR@1: {hr1 - 0.983:+.4f}\n{'='*60}")

    (out_dir / "scores.json").write_text(json.dumps({
        "cell": args.cell, "n_queries": len(queries), "n_hits": hits,
        "hr@1": hr1, "baseline_hr@1": 0.983, "delta_hr@1": hr1 - 0.983,
        "ckpt": "NMFP-triplet ckpt-100 (Araz et al. ISMIR 2025)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
