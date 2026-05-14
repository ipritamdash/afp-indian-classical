"""Move 2: Hubness post-processing on existing NAFP ckpt-10 embeddings.

Directly targets the same-artist failure mode (14/17 misses at 1s) by reweighting
similarity scores to reduce hub-direction inflation. No retraining, no encoder change.

Three methods evaluated:
  1. Inverted Softmax (Smith et al. 2017) — divide each candidate's similarity by
     softmax-sum of similarities to a random sample of reference points.
  2. Mutual k-NN (Schnitzer & Flexer 2015) — only count a candidate as a match if
     the query is reciprocally in the candidate's top-k.
  3. Cross-domain Similarity Local Scaling / CSLS (Conneau et al. 2018) — corrects
     similarity by subtracting mean-similarity to neighbors on both sides.

Operates on:
  - data/results/nafp/saraga_only_main/ref_embs.mm  (690414, 128)
  - data/results/nafp/saraga_only_main/ref_segment_lookup.parquet
  - Query embeddings: re-encoded via existing NAFP encoder (same as intervention2)

Output: data/results/nafp/hubness_postproc/<cell>/query_results_<method>.parquet
        Plus HR@1 comparison vs baseline (0.983).
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
NAFP_DIR = REPO / "data/results/nafp/saraga_only_main"
OUT_DIR = REPO / "data/results/nafp/hubness_postproc"


def setup_log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(__name__)


def build_nafp_encoder(log):
    """Same as intervention2/sweep.py — Mac eager NAFP ckpt-10."""
    sys.path.insert(0, str(REPO / "scripts/nafp/upstream"))
    import yaml, tensorflow as tf
    tf.keras.backend.clear_session()
    with open(REPO / "scripts/nafp/upstream/config/default.yaml") as f:
        cfg = yaml.safe_load(f)
    from model.generate import build_fp
    m_pre, m_fp = build_fp(cfg)
    _ = m_fp(m_pre(tf.zeros((1, 1, 8000), dtype=tf.float32))).numpy()
    ckpt = REPO / "data/results/nafp/kaggle_output/logs/checkpoint"
    checkpoint = tf.train.Checkpoint(model=m_fp)
    status = checkpoint.restore(str(ckpt / "pipeline" / "ckpt-10"))
    status.expect_partial()
    log.info("[encoder] NAFP ckpt-10 loaded")
    return m_pre, m_fp, cfg


def encode_query(audio, m_pre, m_fp, *, win=8000, hop=4000):
    """NAFP query encoding — same as intervention2/sweep.py."""
    import tensorflow as tf
    n = audio.shape[0]
    if n < win:
        return np.empty((0, 128), dtype=np.float32)
    n_seg = (n - win) // hop + 1
    segs = np.empty((n_seg, 1, win), dtype=np.float32)
    for i in range(n_seg):
        segs[i, 0, :] = audio[i * hop:i * hop + win]
    emb = m_fp(m_pre(tf.constant(segs))).numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-12, None)


def sequence_search_with_postproc(q_emb, embs, *,
                                  method: str, k_probe: int, top_k: int,
                                  inv_softmax_anchors: np.ndarray = None,
                                  inv_softmax_tau: float = 0.05,
                                  mutual_k: int = 10,
                                  csls_k: int = 10,
                                  faiss_idx=None):
    """Sequence-search NAFP-style, then re-rank with hubness method.

    method ∈ {'baseline', 'inv_softmax', 'csls'}
    (mutual_kNN omitted — degenerate at L=1 because we'd need top-k from candidate
    BACK to a query pool, which doesn't exist in our 1-query-at-a-time setting.)
    """
    L = q_emb.shape[0]
    # FAISS to get candidates (same as baseline)
    _, I = faiss_idx.search(q_emb.astype(np.float32), k_probe)
    for off in range(L):
        I[off, :] -= off
    cands = np.unique(I[I >= 0])
    cands = cands[cands + L <= embs.shape[0]]
    if len(cands) == 0:
        return []

    # Compute sequence-level baseline similarities
    base_scores = np.empty(len(cands), dtype=np.float32)
    for ci, cid in enumerate(cands):
        ref_chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        base_scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, ref_chunk)))

    if method == "baseline":
        scores = base_scores
    elif method == "inv_softmax":
        # For each candidate's start segment, similarity to the candidate cohort
        # divided by softmax of similarities to a fixed anchor pool.
        # anchor pool = random N_ANCHORS embeddings from the index
        cand_starts = embs[cands]  # (n_cands, 128)
        q_centroid = q_emb.mean(axis=0, keepdims=True)  # (1, 128)  for query side
        # Query→anchor similarities
        q_to_anchors = (q_centroid @ inv_softmax_anchors.T)[0]  # (n_anchors,)
        # Apply softmax with temperature
        norm_q = np.sum(np.exp(q_to_anchors / inv_softmax_tau))
        # Adjusted = base / norm_q (lower for queries with high mean sim — hubness-corrected)
        scores = base_scores / max(norm_q, 1e-12) * 1e3  # scale for numerical sanity
    elif method == "csls":
        # CSLS: sim(x,y) - 0.5*[mean_top_k(x, *) + mean_top_k(*, y)]
        # We approximate "*" using the existing FAISS candidate pool for x (query side)
        # and a precomputed neighbor-mean for y (ref side, via the anchor pool).
        # For ref-side neighbor mean: top-csls_k similarities of candidate to anchors
        cand_starts = embs[cands]  # (n_cands, 128)
        # ref_neighbor_means[i] = mean of top csls_k cosines of cand_starts[i] to anchors
        ref_to_anchors = cand_starts @ inv_softmax_anchors.T  # (n_cands, n_anchors)
        ref_topk = np.partition(ref_to_anchors, -csls_k, axis=1)[:, -csls_k:]
        ref_neighbor_means = ref_topk.mean(axis=1)  # (n_cands,)
        # Query neighbor mean: mean top csls_k cosines of q_emb mean to anchors
        q_centroid = q_emb.mean(axis=0)
        q_to_anchors = inv_softmax_anchors @ q_centroid  # (n_anchors,)
        q_neighbor_mean = np.sort(q_to_anchors)[-csls_k:].mean()
        scores = base_scores - 0.5 * (q_neighbor_mean + ref_neighbor_means)
    else:
        raise ValueError(method)

    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="main_1s")
    ap.add_argument("--queries", nargs="+", default=[
        "data/manifests/hindustani/queries_1s.csv",
        "data/manifests/carnatic/queries_1s.csv"])
    ap.add_argument("--method", choices=["baseline", "inv_softmax", "csls", "all"],
                    default="all")
    ap.add_argument("--n-anchors", type=int, default=4096,
                    help="number of random ref segments to use as hub-correction anchors")
    ap.add_argument("--anchor-seed", type=int, default=20260513)
    args = ap.parse_args()

    log = setup_log()
    out_dir = OUT_DIR / args.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings + lookup
    lookup = pd.read_parquet(NAFP_DIR / "ref_segment_lookup.parquet")
    n_total = len(lookup)
    embs = np.memmap(NAFP_DIR / "ref_embs.mm", dtype=np.float32, mode="r",
                     shape=(n_total, 128))
    log.info(f"[load] {n_total:,} segments")

    # Pick anchor pool (random subset of refs)
    rng = np.random.default_rng(args.anchor_seed)
    anchor_idx = rng.choice(n_total, size=args.n_anchors, replace=False)
    anchors = np.asarray(embs[anchor_idx], dtype=np.float32)
    log.info(f"[anchors] {args.n_anchors} random anchor embeddings sampled (seed={args.anchor_seed})")

    # FAISS index for candidate retrieval
    import faiss
    faiss_idx = faiss.IndexFlatIP(128)
    faiss_idx.add(np.ascontiguousarray(embs, dtype=np.float32))

    # NAFP encoder
    m_pre, m_fp, cfg = build_nafp_encoder(log)
    fs = int(cfg["MODEL"]["FS"]); win = int(cfg["MODEL"]["DUR"] * fs); hop = win // 2

    # Queries
    import librosa
    queries = pd.concat([pd.read_csv(REPO / q) for q in args.queries], ignore_index=True)
    truth = dict(zip(queries.query_id, queries.ref_id))
    lookup_idx = lookup.set_index("global_idx")
    log.info(f"[queries] {len(queries)}")

    methods = ["baseline", "inv_softmax", "csls"] if args.method == "all" else [args.method]
    method_rows = {m: [] for m in methods}

    t0 = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        qid = q["query_id"]
        try:
            audio, _ = librosa.load(q["audio_path"], sr=fs, mono=True)
        except Exception:
            for m in methods:
                method_rows[m].append({"query_id": qid, "rank": 0,
                                       "predicted_ref_id": None, "nafp_score": None})
            continue
        q_emb = encode_query(audio.astype(np.float32), m_pre, m_fp, win=win, hop=hop)
        if q_emb.shape[0] == 0:
            for m in methods:
                method_rows[m].append({"query_id": qid, "rank": 0,
                                       "predicted_ref_id": None, "nafp_score": None})
            continue
        for m in methods:
            hits = sequence_search_with_postproc(
                q_emb, embs, method=m, k_probe=100, top_k=10,
                inv_softmax_anchors=anchors,
                faiss_idx=faiss_idx,
            )
            for rank, (cid, score) in enumerate(hits, start=1):
                info = lookup_idx.loc[cid]
                method_rows[m].append({
                    "query_id": qid, "rank": rank,
                    "predicted_ref_id": str(info["ref_id"]),
                    "predicted_segment_id": int(cid),
                    "nafp_score": float(score),
                })
        if (qi + 1) % 100 == 0:
            log.info(f"[q] {qi+1}/{len(queries)} elapsed={time.time()-t0:.0f}s")

    # Write + score
    results = {}
    for m in methods:
        df = pd.DataFrame(method_rows[m])
        df.to_parquet(out_dir / f"query_results_{m}.parquet", index=False)
        top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
        hits = sum(1 for qid in queries.query_id if truth[qid] == top1.get(qid))
        hr1 = hits / len(queries)
        results[m] = {"n_hits": hits, "hr@1": hr1, "delta_vs_base": hr1 - 0.983}
        log.info(f"[result] {m:15s} HR@1={hr1:.4f}  ({hits}/{len(queries)})  Δ={hr1-0.983:+.4f}")

    (out_dir / "scores.json").write_text(json.dumps({
        "cell": args.cell,
        "n_queries": len(queries),
        "baseline_hr@1": 0.983,
        "results": results,
        "anchor_pool_size": args.n_anchors,
        "anchor_seed": args.anchor_seed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
