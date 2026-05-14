"""NAFP inference on Kaggle GPU — single-shot for the Saraga benchmark.

Inputs (Kaggle datasets attached):
  - aboutpritam/nafp-infer-bundle  → ckpt-10, NAFP source, manifests
  - aboutpritam/nafp-wheels         → kapre 0.3.7 + tf_keras 2.20.x offline wheels
  - Saraga 1.5: downloaded via mirdata at runtime from Zenodo (enable_internet=true)

Output (to /kaggle/working/):
  - ref_embs.mm                       — (n_segments, 128) float32 memmap, contiguous
  - ref_segment_lookup.parquet         — (global_idx, ref_id, segment_idx, segment_start_sec)
  - query_results.parquet              — (query_id, rank, predicted_ref_id, …) — same schema as the 3 classical runners
  - index_log.json, query_log.json     — diagnostic
  - manifest_run.json                  — version, GPU details, timings

Fail-fast design (same as v5 training kernel):
  - sys.exit(7) if no GPU
  - sys.exit(8) if /GPU:0 matmul > 200 ms
  - Bit-deterministic on a given Kaggle image: TF op_determinism + intra-op threads pinned
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── 0) Environment — MUST be set BEFORE TF import ──────────────────────────
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

WORKING = Path("/kaggle/working")
SARAGA_ROOT = Path("/kaggle/working/saraga")  # mirdata downloads here


def run(cmd, **kw):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


# ── 1) Discover input datasets ──────────────────────────────────────────────
def find_input(name: str) -> Path:
    root = Path("/kaggle/input")
    candidates = list(root.glob(f"{name}*")) + list(root.glob(f"datasets/aboutpritam/{name}"))
    for c in candidates:
        if c.is_dir():
            return c
    print(f"FATAL: input dataset {name!r} not found under /kaggle/input/", flush=True)
    print("contents:", flush=True)
    for p in root.rglob("*"):
        if p.is_dir() and p.depth <= 3:
            print(f"  {p}", flush=True)
    raise SystemExit(2)


def main() -> int:
    t_start = time.time()
    print(f"=== NAFP inference @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    bundle = find_input("nafp-infer-bundle")
    wheels = find_input("nafp-wheels")
    print(f"bundle = {bundle}", flush=True)
    print(f"wheels = {wheels}", flush=True)

    # ── 2) Offline install kapre + tf_keras + faiss + mirdata + huggingface_hub
    print("\n[deps] installing", flush=True)
    pip_args = [sys.executable, "-m", "pip", "install", "-q",
                "--no-index", "--find-links", str(wheels),
                "kapre", "tf_keras"]
    run(pip_args, timeout=180)
    # internet-installs for everything else (faiss; we no longer need mirdata since
    # Saraga refs come from our private Kaggle dataset)
    run([sys.executable, "-m", "pip", "install", "-q",
         "faiss-cpu>=1.7"], timeout=300)

    # ── 3) Stage NAFP source + add to path ──────────────────────────────────
    NAFP_REPO = WORKING / "neural-audio-fp"
    if NAFP_REPO.exists():
        shutil.rmtree(NAFP_REPO)
    shutil.copytree(bundle / "nafp_upstream", NAFP_REPO)
    sys.path.insert(0, str(NAFP_REPO))
    print(f"copied NAFP source → {NAFP_REPO}", flush=True)

    # Patch the CosineDecay alias used by trainer.py (not needed for inference but safe)
    trainer = NAFP_REPO / "model" / "trainer.py"
    if trainer.exists():
        src = trainer.read_text()
        new = src.replace("tf.keras.experimental.CosineDecay",
                          "tf.keras.optimizers.schedules.CosineDecay")
        if new != src:
            trainer.write_text(new)

    # ── 4) HARD GPU + matmul assertion (mirrors v5 training kernel) ─────────
    import tensorflow as tf
    import numpy as np
    print(f"TF: {tf.__version__}", flush=True)

    # Pin seeds BEFORE any random op (TF_DETERMINISTIC_OPS=1 requires it)
    tf.keras.utils.set_random_seed(20260512)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"(enable_op_determinism failed: {e})", flush=True)

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible: {len(gpus)}: {gpus}", flush=True)
    if not gpus:
        print("FATAL[7]: no GPU visible — kernel-metadata says enable_gpu=true", flush=True)
        return 7
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print(f"  (memory growth set failed: {e})", flush=True)

    # Use np.random (with our seeded global state via set_random_seed) for the
    # warmup tensors — avoids the `tf.random.normal needs seed` requirement
    # under TF_DETERMINISTIC_OPS without breaking reproducibility (we control
    # the np random state via set_random_seed which seeds numpy too).
    print("\nGPU compute check (2048×2048 matmul)…", flush=True)
    a_np = np.random.randn(2048, 2048).astype(np.float32)
    b_np = np.random.randn(2048, 2048).astype(np.float32)
    with tf.device("/GPU:0"):
        a = tf.constant(a_np)
        b = tf.constant(b_np)
        _ = tf.matmul(a, b).numpy()  # warmup
        t0 = time.time()
        c = tf.matmul(a, b); _ = c.numpy()
        gpu_matmul_ms = (time.time() - t0) * 1000
    print(f"GPU matmul: {gpu_matmul_ms:.1f} ms (healthy < 30 ms; CPU fallback > 500 ms)", flush=True)
    if gpu_matmul_ms > 200:
        print(f"FATAL[8]: matmul too slow — TF on CPU despite GPU listing", flush=True)
        return 8

    # ── 6) Build NAFP encoder + load ckpt-10 ───────────────────────────────
    import yaml
    from model.generate import build_fp, load_checkpoint  # noqa: E402

    cfg = yaml.safe_load((NAFP_REPO / "config" / "default.yaml").read_text())
    m_pre, m_fp = build_fp(cfg)
    ckpt_root = str(bundle / "ckpt")
    actual_idx = load_checkpoint(ckpt_root, "pipeline", 10, m_fp)
    print(f"\nencoder ready; restored ckpt-{actual_idx}", flush=True)

    fs = int(cfg["MODEL"]["FS"])             # 8000
    dur = float(cfg["MODEL"]["DUR"])         # 1.0
    hop = float(cfg["MODEL"]["HOP"])         # 0.5
    win_samples = int(fs * dur)              # 8000
    hop_samples = int(fs * hop)              # 4000
    emb_dim = int(cfg["MODEL"]["EMB_SZ"])    # 128
    print(f"cfg: fs={fs}  win={win_samples}  hop={hop_samples}  emb_dim={emb_dim}", flush=True)

    # Warmup eager forward pass — concretizes all sublayer shapes (DivEncLayer in
    # particular needs this; under @tf.function with None batch-dim its internal
    # reshape fails with "Failed to convert elements of [None, 128, -1] to Tensor").
    print("warmup eager forward pass to build sublayers…", flush=True)
    _ = m_fp(m_pre(tf.zeros((1, 1, win_samples), dtype=tf.float32))).numpy()
    print("warmup done", flush=True)

    # Use plain eager mode — @tf.function fails on DivEncLayer reshape with None
    # batch dim on this TF version. Eager is slower but reliable.
    def encode_batch(x):
        return m_fp(m_pre(x))

    # ── 7) Load manifests (combined H + C) ─────────────────────────────────
    import pandas as pd
    refs_h = pd.read_csv(bundle / "manifests" / "hindustani_refs.csv")
    refs_c = pd.read_csv(bundle / "manifests" / "carnatic_refs.csv")
    refs_df = pd.concat([refs_h, refs_c], ignore_index=True)
    queries_h = pd.read_csv(bundle / "manifests" / "hindustani_queries.csv")
    queries_c = pd.read_csv(bundle / "manifests" / "carnatic_queries.csv")
    queries_df = pd.concat([queries_h, queries_c], ignore_index=True)
    print(f"manifests: {len(refs_df)} refs, {len(queries_df)} queries", flush=True)

    # ── 8) Resolve refs from /kaggle/input/saraga-refs-flat/ ──────────────
    # Kaggle may have escaped special chars in filenames (e.g. `&`) during upload,
    # so do an exact match first, then a glob fallback that handles those cases.
    import glob as _glob
    saraga_root = find_input("saraga-refs-flat")
    print(f"\nsaraga refs root = {saraga_root}", flush=True)

    # Pre-list all .mp3s once for fast lookup
    all_mp3s = list(saraga_root.glob("*.mp3"))
    by_stem = {p.stem: p for p in all_mp3s}
    print(f"  {len(all_mp3s)} mp3 files on dataset", flush=True)

    audio_map: dict[str, str] = {}
    missing = []
    for rid in refs_df["ref_id"]:
        # 1) exact match
        if rid in by_stem:
            audio_map[rid] = str(by_stem[rid])
            continue
        # 2) explicit substitutions for known Kaggle-unsafe characters
        #    (Kaggle filename escaping can lose `&`; our local rename uses `_and_`)
        for src, dst in [("_&_", "_and_"), ("&", "and"), (" ", "_")]:
            if src in rid:
                alt = rid.replace(src, dst)
                if alt in by_stem:
                    audio_map[rid] = str(by_stem[alt])
                    break
        if rid in audio_map:
            continue
        # 3) glob fallback — replace each non-alphanumeric with a wildcard
        import re
        pattern = re.sub(r"[^A-Za-z0-9_]", "?", rid) + ".mp3"
        matches = sorted(saraga_root.glob(pattern))
        if len(matches) == 1:
            audio_map[rid] = str(matches[0])
        elif len(matches) > 1:
            print(f"  ambiguous match for {rid!r}: {[m.name for m in matches[:3]]} — taking first", flush=True)
            audio_map[rid] = str(matches[0])
        else:
            missing.append(rid)

    print(f"audio_map: {len(audio_map)}/{len(refs_df)} refs resolved (missing: {len(missing)})", flush=True)
    if missing:
        print(f"  missing examples: {missing[:5]}", flush=True)
    # Do not fatal — drop the missing refs and proceed. Document in manifest_run.json.
    refs_df = refs_df[refs_df["ref_id"].isin(audio_map.keys())].reset_index(drop=True)
    print(f"  refs_df trimmed to {len(refs_df)} resolved refs", flush=True)

    # ── 10) Index refs ─────────────────────────────────────────────────────
    import librosa
    import numpy as np

    print("\n=== INDEX refs ===", flush=True)
    # First pass: count
    n_per_ref = []
    for _, r in refs_df.iterrows():
        d = float(r["duration_sec"])
        n_seg = max(0, int(d * fs - win_samples) // hop_samples + 1)
        n_per_ref.append(n_seg)
    n_total = int(sum(n_per_ref))
    print(f"total segments: {n_total:,}", flush=True)

    emb_path = WORKING / "ref_embs.mm"
    embs = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n_total, emb_dim))
    lookup_rows = []
    offset = 0
    BATCH = 512
    t0_total = time.time()
    for i, (_, r) in enumerate(refs_df.iterrows()):
        ref_id = r["ref_id"]
        if ref_id not in audio_map:
            continue
        n_seg_expected = n_per_ref[i]
        if n_seg_expected == 0:
            continue
        t0 = time.time()
        audio, _ = librosa.load(audio_map[ref_id], sr=fs, mono=True)
        audio = audio.astype(np.float32, copy=False)
        n_avail = (len(audio) - win_samples) // hop_samples + 1
        if n_avail <= 0:
            print(f"  [skip] {ref_id}: only {len(audio)} samples", flush=True)
            continue
        n_seg = min(n_avail, n_seg_expected)
        # build (n_seg, 1, win) tensor via numpy view + copy
        segs = np.empty((n_seg, 1, win_samples), dtype=np.float32)
        for j in range(n_seg):
            s = j * hop_samples
            segs[j, 0, :] = audio[s:s + win_samples]
        # encode in batches
        out_emb = np.empty((n_seg, emb_dim), dtype=np.float32)
        for s in range(0, n_seg, BATCH):
            e = min(s + BATCH, n_seg)
            xb = tf.constant(segs[s:e], dtype=tf.float32)
            out_emb[s:e] = encode_batch(xb).numpy()
        # defensive L2-renorm
        norms = np.linalg.norm(out_emb, axis=1, keepdims=True)
        out_emb /= np.clip(norms, 1e-12, None)
        embs[offset:offset + n_seg, :] = out_emb
        for j in range(n_seg):
            lookup_rows.append({
                "global_idx": offset + j,
                "ref_id": ref_id,
                "segment_idx": j,
                "segment_start_sec": j * (hop_samples / fs),
            })
        offset += n_seg
        if (i + 1) % 25 == 0 or i < 5:
            print(f"  [index] {i+1}/{len(refs_df)}  {ref_id}  segs={n_seg}  "
                  f"{time.time()-t0:.2f}s  total_elapsed={time.time()-t0_total:.1f}s", flush=True)

    if offset != n_total:
        print(f"truncating memmap: actual={offset} predicted={n_total}", flush=True)
        embs.flush(); del embs
        embs = np.memmap(emb_path, dtype="float32", mode="r+", shape=(offset, emb_dim))
    embs.flush()
    lookup_df = pd.DataFrame(lookup_rows)
    lookup_df.to_parquet(WORKING / "ref_segment_lookup.parquet", index=False)
    idx_stats = {
        "n_refs": int(len(refs_df)),
        "n_segments_total": int(offset),
        "ref_embs_mb": round(emb_path.stat().st_size / 1e6, 2),
        "index_seconds": round(time.time() - t0_total, 1),
        "fs": fs, "win_samples": win_samples, "hop_samples": hop_samples,
        "checkpoint_index": int(actual_idx),
    }
    (WORKING / "index_log.json").write_text(json.dumps(idx_stats, indent=2))
    print(f"\n=== INDEX done: {idx_stats} ===", flush=True)

    # ── 11) Build FAISS index ──────────────────────────────────────────────
    import faiss
    print("\n[faiss] building IndexFlatIP", flush=True)
    embs_ro = np.memmap(emb_path, dtype="float32", mode="r", shape=(offset, emb_dim))
    index = faiss.IndexFlatIP(emb_dim)
    index.add(np.ascontiguousarray(embs_ro, dtype=np.float32))
    print(f"[faiss] ntotal={index.ntotal}", flush=True)

    # ── 12) Run queries — encode + sequence-search ─────────────────────────
    print(f"\n=== QUERY {len(queries_df)} ===", flush=True)
    lookup_by_idx = lookup_df.set_index("global_idx")[["ref_id", "segment_start_sec"]]
    K_PROBE, TOP_K = 20, 10

    def sequence_search(q_emb: np.ndarray):
        L = q_emb.shape[0]
        _, I = index.search(q_emb.astype(np.float32, copy=False), K_PROBE)
        for off in range(L):
            I[off, :] -= off
        cands = np.unique(I[I >= 0])
        cands = cands[cands + L <= offset]
        if len(cands) == 0:
            return []
        scores = np.empty(len(cands), dtype=np.float32)
        for ci, cid in enumerate(cands):
            ref_chunk = np.asarray(embs_ro[cid:cid + L])
            scores[ci] = float(np.mean(np.einsum("ij,ij->i", q_emb, ref_chunk)))
        order = np.argsort(-scores)[:TOP_K]
        return [(int(cands[o]), float(scores[o])) for o in order]

    rows = []
    latencies = []
    n_no_match = 0
    # Re-cut queries from already-supplied audio_path → but those paths are LOCAL and not on Kaggle.
    # Instead we re-cut from Saraga audio at the manifest's offset_sec for each query.
    for qi, (_, q) in enumerate(queries_df.iterrows()):
        query_id = q["query_id"]; ref_id = q["ref_id"]
        offset_sec = float(q["offset_sec"]); length_sec = float(q["length_sec"])
        if ref_id not in audio_map:
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": None})
            continue
        # Load the precise query window directly from source mp3
        import soundfile as sf
        with sf.SoundFile(audio_map[ref_id]) as f:
            sr_src = f.samplerate
            start = int(round(offset_sec * sr_src))
            frames = int(round(length_sec * sr_src))
            f.seek(start)
            data = f.read(frames=frames, dtype="float32", always_2d=True)
        if data.shape[1] > 1: data = data.mean(axis=1)
        else: data = data[:, 0]
        if sr_src != fs:
            data = librosa.resample(data, orig_sr=sr_src, target_sr=fs, res_type="soxr_hq")
        n_avail = (len(data) - win_samples) // hop_samples + 1
        if n_avail <= 0:
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": None})
            continue
        segs = np.empty((n_avail, 1, win_samples), dtype=np.float32)
        for j in range(n_avail):
            s = j * hop_samples
            segs[j, 0, :] = data[s:s + win_samples]
        t0 = time.time()
        q_emb = encode_batch(tf.constant(segs, dtype=tf.float32)).numpy()
        norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_emb /= np.clip(norms, 1e-12, None)
        hits = sequence_search(q_emb)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        if not hits:
            n_no_match += 1
            rows.append({"query_id": query_id, "rank": 0, "predicted_ref_id": None,
                         "predicted_segment_id": None, "match_count": 0,
                         "ref_start": None, "ref_stop": None,
                         "query_start": None, "query_stop": None,
                         "nafp_score": None, "latency_ms": latency_ms})
            continue
        L = q_emb.shape[0]
        seq_dur = (L - 1) * (hop_samples / fs) + (win_samples / fs)
        for rank, (cid, score) in enumerate(hits, start=1):
            rid = lookup_by_idx.at[cid, "ref_id"]
            rs = float(lookup_by_idx.at[cid, "segment_start_sec"])
            rows.append({
                "query_id": query_id, "rank": rank,
                "predicted_ref_id": rid, "predicted_segment_id": cid,
                "match_count": L,
                "ref_start": rs, "ref_stop": rs + seq_dur,
                "query_start": 0.0, "query_stop": seq_dur,
                "nafp_score": score, "latency_ms": latency_ms,
            })
        if (qi + 1) % 100 == 0:
            print(f"  [query] {qi+1}/{len(queries_df)}  last={query_id}  "
                  f"top1={hits[0][1]:.4f}  latency={latency_ms:.0f}ms", flush=True)

    results_df = pd.DataFrame(rows)
    results_df.to_parquet(WORKING / "query_results.parquet", index=False)
    q_stats = {
        "n_queries": int(len(queries_df)),
        "n_no_match": int(n_no_match),
        "frac_no_match": round(n_no_match / max(1, len(queries_df)), 4),
        "latency_ms_p50": float(np.median(latencies)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "k_probe": K_PROBE, "top_k": TOP_K,
    }
    (WORKING / "query_log.json").write_text(json.dumps(q_stats, indent=2))
    print(f"\n=== QUERY done: {q_stats} ===", flush=True)

    manifest = {
        "version": "v1-inference",
        "tf_version": tf.__version__,
        "gpu_matmul_ms": round(gpu_matmul_ms, 1),
        "index_seconds": idx_stats["index_seconds"],
        "total_seconds": round(time.time() - t_start, 1),
        "ckpt_index": int(actual_idx),
        "n_refs": idx_stats["n_refs"],
        "n_segments_total": idx_stats["n_segments_total"],
        "n_queries": q_stats["n_queries"],
    }
    (WORKING / "manifest_run.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
