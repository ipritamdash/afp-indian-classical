"""Evaluate NAFP+NMFP-recipe ckpt-10 on Saraga (per pre-registered PROTOCOL).

Loads:
  - recipe_v2.yaml config (F_MIN=160, TR_SEG_MODE=random_oneshot)
  - ckpt-10 from Colab training (10 epochs, NMFP recipe applied)

Outputs to data/results/nafp/recipe_v2_10ep/:
  - ref_embs.mm (357 refs × 128-D)
  - ref_segment_lookup.parquet
  - query_results_<cell>.parquet (per cell)
  - scores_<cell>.json (per cell)
"""
from __future__ import annotations
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# CUDA_VISIBLE_DEVICES NOT setdefault'd here — caller controls.
# CPU run: launch with `CUDA_VISIBLE_DEVICES=-1 uv run python ...`
# Metal run: launch with /tmp/venv_metal/bin/python (Metal sees its own GPU regardless)

import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# These are defaults; --ckpt-dir / --cfg / --out-dir override.
CKPT_DIR = REPO / "data/results/nafp/recipe_v2_10ep/checkpoint"
CFG_PATH = REPO / "data/results/nafp/recipe_v2_10ep/recipe_v2.yaml"
DEFAULT_OUT_DIR = REPO / "data/results/nafp/recipe_v2_10ep"
OUT_DIR = DEFAULT_OUT_DIR
CKPT_NAME = "ckpt-10"  # set by main() — name of checkpoint file inside CKPT_DIR


def setup_log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(__name__)


def build_encoder(log):
    sys.path.insert(0, str(REPO / "scripts/nafp/upstream"))
    import tensorflow as tf
    import yaml
    tf.keras.backend.clear_session()
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    log.info(f"[cfg] F_MIN={cfg['MODEL']['F_MIN']}  EMB_SZ={cfg['MODEL']['EMB_SZ']}")
    from model.generate import build_fp
    m_pre, m_fp = build_fp(cfg)
    _ = m_fp(m_pre(tf.zeros((1, 1, 8000), dtype=tf.float32))).numpy()  # build sublayers

    checkpoint = tf.train.Checkpoint(model=m_fp)
    status = checkpoint.restore(str(CKPT_DIR / CKPT_NAME))
    status.expect_partial()
    log.info(f"[ckpt] Restored {CKPT_NAME} from {CKPT_DIR}")
    return m_pre, m_fp, cfg


def slice_segments(audio, win, hop):
    n = audio.shape[0]
    if n < win:
        return np.empty((0, 1, win), dtype=np.float32)
    n_seg = (n - win) // hop + 1
    out = np.empty((n_seg, 1, win), dtype=np.float32)
    for i in range(n_seg):
        out[i, 0, :] = audio[i * hop:i * hop + win]
    return out


def encode_audio(audio, m_pre, m_fp, *, win, hop, batch=128):
    import tensorflow as tf
    segs = slice_segments(audio.astype(np.float32), win, hop)
    n = segs.shape[0]
    if n == 0:
        return np.empty((0, 128), dtype=np.float32)
    out = np.empty((n, 128), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        emb = m_fp(m_pre(tf.constant(segs[s:e]))).numpy()
        out[s:e] = emb
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-12, None)


def sanity(m_pre, m_fp, log, fs, win, hop):
    import librosa
    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)
    ref = refs[refs.ref_id == "hindustani_0_Raag_Shree"].iloc[0]
    audio, _ = librosa.load(ref["audio_path"], sr=fs, mono=True)
    a = audio[1028 * hop:1028 * hop + win].astype(np.float32)
    b = audio[2000 * hop:2000 * hop + win].astype(np.float32)
    ea = encode_audio(a, m_pre, m_fp, win=win, hop=hop)
    eb = encode_audio(b, m_pre, m_fp, win=win, hop=hop)
    log.info(f"[sanity] ‖ea‖={np.linalg.norm(ea[0]):.4f}  cos(a,b)={float(ea[0] @ eb[0]):.4f}")
    return 0.0 < float(ea[0] @ eb[0]) < 0.999


def index_refs(m_pre, m_fp, log, fs, win, hop):
    import librosa
    refs = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/refs.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/refs.csv"),
    ], ignore_index=True)
    n_per_ref = [max(0, (int(float(r["duration_sec"]) * fs) - win) // hop + 1)
                 for _, r in refs.iterrows()]
    n_total = sum(n_per_ref)
    log.info(f"[index] estimated {n_total:,} segments over {len(refs)} refs")
    emb_path = OUT_DIR / "ref_embs.mm"
    embs = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_total, 128))
    rows, off, t0 = [], 0, time.time()
    for i, (_, r) in enumerate(refs.iterrows()):
        try:
            audio, _ = librosa.load(r["audio_path"], sr=fs, mono=True)
        except Exception as exc:
            log.error(f"[index] load failed for {r.ref_id}: {exc}")
            continue
        emb = encode_audio(audio, m_pre, m_fp, win=win, hop=hop)
        n = min(emb.shape[0], n_per_ref[i])
        if n == 0:
            continue
        embs[off:off + n] = emb[:n]
        for si in range(n):
            rows.append({"global_idx": off + si, "ref_id": r.ref_id,
                         "segment_idx": si, "segment_start_sec": si * (hop / fs)})
        off += n
        if (i + 1) % 25 == 0 or i == len(refs) - 1:
            log.info(f"[index] {i+1}/{len(refs)}  {r.ref_id}  +{n}  total={off:,}  elapsed={time.time()-t0:.0f}s")
    if off != n_total:
        embs.flush(); del embs
        embs = np.memmap(emb_path, dtype=np.float32, mode="r+", shape=(off, 128))
    embs.flush()
    lookup = pd.DataFrame(rows)
    lookup.to_parquet(OUT_DIR / "ref_segment_lookup.parquet", index=False)
    log.info(f"[index] DONE — {off:,} segs in {time.time()-t0:.0f}s")
    return embs, lookup, off


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
        chunk = np.asarray(embs[cid:cid + L], dtype=np.float32)
        scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, chunk)))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(cands[o]), float(scores[o])) for o in order]


def eval_cell(m_pre, m_fp, embs, lookup, fs, win, hop, cell, h_q, c_q, log):
    import librosa, faiss
    out_cell = OUT_DIR / cell
    out_cell.mkdir(exist_ok=True)
    queries = pd.concat([pd.read_csv(REPO / h_q), pd.read_csv(REPO / c_q)], ignore_index=True)
    truth = dict(zip(queries.query_id, queries.ref_id))
    log.info(f"[{cell}] {len(queries)} queries")
    index = faiss.IndexFlatIP(128)
    index.add(np.ascontiguousarray(embs, dtype=np.float32))
    lookup_idx = lookup.set_index("global_idx")
    rows, t0 = [], time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        try:
            audio, _ = librosa.load(q["audio_path"], sr=fs, mono=True)
        except Exception:
            rows.append({"query_id": q.query_id, "rank": 0, "predicted_ref_id": None,
                         "nafp_score": None}); continue
        q_emb = encode_audio(audio, m_pre, m_fp, win=win, hop=hop)
        if q_emb.shape[0] == 0:
            rows.append({"query_id": q.query_id, "rank": 0, "predicted_ref_id": None,
                         "nafp_score": None}); continue
        hits = sequence_search(q_emb, index, embs)
        for rank, (cid, score) in enumerate(hits, start=1):
            info = lookup_idx.loc[cid]
            rows.append({"query_id": q.query_id, "rank": rank,
                         "predicted_ref_id": str(info["ref_id"]),
                         "predicted_segment_id": int(cid),
                         "nafp_score": float(score)})
        if (qi + 1) % 100 == 0:
            log.info(f"[{cell}] {qi+1}/{len(queries)} elapsed={time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(out_cell / "query_results.parquet", index=False)
    top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
    hits = sum(1 for qid in queries.query_id if truth[qid] == top1.get(qid))
    hr1 = hits / len(queries)
    log.info(f"[{cell}] HR@1 = {hr1:.4f}  ({hits}/{len(queries)})")
    (out_cell / "scores.json").write_text(json.dumps({
        "cell": cell, "n_queries": len(queries), "n_hits": hits, "hr@1": hr1,
        "ckpt": "NAFP-NMFPrecipe-ckpt-10",
    }, indent=2))
    return hr1, hits, len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["main_1s"],
                    help="cells to eval (e.g., main_1s main_3s main_5s main_10s)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: data/results/nafp/recipe_v2_10ep)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="checkpoint directory (e.g., data/results/nafp/recipe_v3_30ep/seed42)")
    ap.add_argument("--ckpt-name", default=None,
                    help="checkpoint file basename (e.g., ckpt-30). Default ckpt-10.")
    ap.add_argument("--cfg", default=None,
                    help="config yaml path (default: recipe_v2 yaml)")
    args = ap.parse_args()
    global OUT_DIR, CKPT_DIR, CKPT_NAME, CFG_PATH
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    if args.ckpt_dir:
        CKPT_DIR = Path(args.ckpt_dir)
    if args.ckpt_name:
        CKPT_NAME = args.ckpt_name
    if args.cfg:
        CFG_PATH = Path(args.cfg)
    log = setup_log()
    log.info(f"[out] writing to {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m_pre, m_fp, cfg = build_encoder(log)
    fs = int(cfg["MODEL"]["FS"]); win = int(cfg["MODEL"]["DUR"] * fs); hop = win // 2
    assert sanity(m_pre, m_fp, log, fs, win, hop), "sanity check failed"
    log.info("[sanity] PASS")

    # Index refs ONCE
    if (OUT_DIR / "ref_embs.mm").exists() and (OUT_DIR / "ref_segment_lookup.parquet").exists():
        lookup = pd.read_parquet(OUT_DIR / "ref_segment_lookup.parquet")
        n = len(lookup)
        embs = np.memmap(OUT_DIR / "ref_embs.mm", dtype=np.float32, mode="r", shape=(n, 128))
        log.info(f"[index] reuse existing — {n:,} segs")
    else:
        embs, lookup, _ = index_refs(m_pre, m_fp, log, fs, win, hop)

    # Cell-to-manifest mapping
    cell_map = {
        "main_1s":   ("data/manifests/hindustani/queries_1s.csv", "data/manifests/carnatic/queries_1s.csv"),
        "main_3s":   ("data/manifests/hindustani/queries_3s.csv", "data/manifests/carnatic/queries_3s.csv"),
        "main_5s":   ("data/manifests/hindustani/queries_5s.csv", "data/manifests/carnatic/queries_5s.csv"),
        "main_10s":  ("data/manifests/hindustani/queries.csv",    "data/manifests/carnatic/queries.csv"),
        "ablation_1s":  ("data/manifests/hindustani/queries_ablation_1s.csv", "data/manifests/carnatic/queries_ablation_1s.csv"),
        "ablation_3s":  ("data/manifests/hindustani/queries_ablation_3s.csv", "data/manifests/carnatic/queries_ablation_3s.csv"),
        "ablation_5s":  ("data/manifests/hindustani/queries_ablation_5s.csv", "data/manifests/carnatic/queries_ablation_5s.csv"),
        "ablation_10s": ("data/manifests/hindustani/queries_ablation.csv",    "data/manifests/carnatic/queries_ablation.csv"),
    }
    for c in args.cells:
        if c not in cell_map:
            log.error(f"unknown cell: {c}"); continue
        h_q, c_q = cell_map[c]
        eval_cell(m_pre, m_fp, embs, lookup, fs, win, hop, c, h_q, c_q, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
