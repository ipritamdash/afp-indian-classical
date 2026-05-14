"""Olaf system runner.

Indexes a reference library and runs every query against it, capturing
top-K matches, per-query latency, and total DB-on-disk size.

Reproducibility contract:
  * Uses the bundled Olaf Python wrapper (CFFI) — no Zig, no ffmpeg
  * Audio loaded externally via librosa (sample-accurate, deterministic)
  * Single-threaded querying (Olaf wrapper restriction: class-level results list)
  * DB location: $HOME/.olaf/db/ (Olaf default, cleared at start of indexing)
  * Records audio_identifier (Jenkins-hashed path) per ref; output rows
    use predicted_ref_id resolved via that map

Outputs:
  <out_dir>/index_log.json     {n_refs, n_indexed, n_failed, store_total_sec,
                                db_size_bytes, samples_loaded, …}
  <out_dir>/query_results.parquet  long format: one row per (query, rank)
    columns: query_id, rank, predicted_ref_id, match_count, ref_start, ref_stop,
             query_start, query_stop, latency_ms
  <out_dir>/query_log.json     {n_queries, latencies_ms_p50/p95/p99, n_with_match, …}

Usage:
  uv run python scripts/systems/olaf_runner.py \\
      --refs    data/manifests/hindustani/refs.csv \\
                data/manifests/carnatic/refs.csv \\
      --queries data/manifests/hindustani/queries.csv \\
                data/manifests/carnatic/queries.csv \\
      --output  data/results/olaf/saraga_only \\
      --top-k   10
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OLAF_DIR = PROJECT_ROOT / "external" / "Olaf"

# Make the Olaf wrapper + CFFI module importable
sys.path.insert(0, str(OLAF_DIR))
sys.path.insert(0, str(OLAF_DIR / "python-wrapper"))

from olaf import Olaf, OlafCommand  # noqa: E402  (must follow sys.path edits)
from olaf_cffi import ffi, lib  # noqa: E402


OLAF_HOME_DB = Path.home() / ".olaf" / "db"


def olaf_identifier(path: str) -> int:
    """Compute Olaf's per-track audio identifier (Jenkins hash) for a path."""
    pb = path.encode("utf-8")
    return int(lib.olaf_db_string_hash(ffi.new("char []", pb), len(pb)))


def clear_db() -> None:
    if OLAF_HOME_DB.exists():
        for p in OLAF_HOME_DB.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    else:
        OLAF_HOME_DB.mkdir(parents=True, exist_ok=True)


def db_size_bytes() -> int:
    if not OLAF_HOME_DB.exists():
        return 0
    total = 0
    for p in OLAF_HOME_DB.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def load_audio_for_olaf(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file at the Olaf sample rate, mono float32, redirecting
    soundfile-first to avoid librosa fallback overhead on MP3."""
    # soundfile + libsndfile handles MP3 on this system (verified Phase 2).
    try:
        with sf.SoundFile(path) as f:
            orig_sr = f.samplerate
            data = f.read(dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if orig_sr != target_sr:
            mono = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
        return mono.astype(np.float32, copy=False)
    except Exception:
        y, _ = librosa.load(path, sr=target_sr, mono=True)
        return y.astype(np.float32, copy=False)


def index_refs(refs_df: pd.DataFrame, log: logging.Logger) -> dict:
    cfg = lib.olaf_config_default()
    target_sr = int(cfg.audioSampleRate)
    log.info(f"clearing Olaf DB at {OLAF_HOME_DB}")
    clear_db()

    n = len(refs_df)
    failed: list[tuple[str, str]] = []
    samples_total = 0
    audio_load_sec = 0.0
    store_sec = 0.0
    audio_id_map: dict[int, str] = {}
    duplicate_id_collisions = 0

    t_outer = time.time()
    for i, (_, row) in enumerate(refs_df.iterrows()):
        ref_id = row["ref_id"]
        path = row["audio_path"]
        try:
            t0 = time.time()
            y = load_audio_for_olaf(path, target_sr)
            audio_load_sec += time.time() - t0
            samples_total += len(y)

            t0 = time.time()
            olaf = Olaf(OlafCommand.STORE, path)
            ident = int(olaf.audio_identifier)
            olaf.do(y=y)
            del olaf  # ensures __del__ writes and closes the DB handle
            store_sec += time.time() - t0

            if ident in audio_id_map and audio_id_map[ident] != ref_id:
                duplicate_id_collisions += 1
                log.warning(f"identifier collision: {ident} already maps to {audio_id_map[ident]}, now {ref_id}")
            audio_id_map[ident] = ref_id
        except Exception as e:
            failed.append((ref_id, repr(e)))
            log.error(f"  STORE FAILED for {ref_id}: {e!r}")

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_outer
            log.info(f"  indexed {i+1}/{n}  (audio_load={audio_load_sec:.1f}s, store={store_sec:.1f}s, total_wall={elapsed:.1f}s)")

    db_size = db_size_bytes()
    return {
        "n_refs": n,
        "n_indexed": n - len(failed),
        "n_failed": len(failed),
        "failed_samples": failed[:10],
        "audio_load_sec": round(audio_load_sec, 2),
        "store_sec": round(store_sec, 2),
        "total_wall_sec": round(time.time() - t_outer, 2),
        "samples_loaded": int(samples_total),
        "db_size_bytes": int(db_size),
        "audio_id_map_size": len(audio_id_map),
        "duplicate_id_collisions": duplicate_id_collisions,
        "audio_id_map": {str(k): v for k, v in audio_id_map.items()},
    }


def run_queries(queries_df: pd.DataFrame, audio_id_map: dict[int, str],
                top_k: int, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    cfg = lib.olaf_config_default()
    target_sr = int(cfg.audioSampleRate)

    rows_out: list[dict] = []
    latencies_ms: list[float] = []
    n_with_match = 0

    n = len(queries_df)
    t_outer = time.time()
    for i, (_, row) in enumerate(queries_df.iterrows()):
        qid = row["query_id"]
        qpath = row["audio_path"]
        try:
            y = load_audio_for_olaf(qpath, target_sr)
            t0 = time.time()
            olaf = Olaf(OlafCommand.QUERY, qpath)
            results = olaf.do(y=y)
            del olaf
            lat = (time.time() - t0) * 1000.0
            latencies_ms.append(lat)

            if not results:
                rows_out.append({
                    "query_id": qid,
                    "rank": 0,
                    "predicted_ref_id": None,
                    "match_count": 0,
                    "ref_start": None,
                    "ref_stop": None,
                    "query_start": None,
                    "query_stop": None,
                    "latency_ms": lat,
                })
                continue

            n_with_match += 1
            # group by matchIdentifier — multiple match windows per reference are
            # possible; we collapse to one row per (query, predicted_ref) using
            # the max matchCount window
            best_per_ref: dict[int, dict] = {}
            for r in results:
                mid = int(r["matchIdentifier"])
                if mid not in best_per_ref or r["matchCount"] > best_per_ref[mid]["matchCount"]:
                    best_per_ref[mid] = r
            sorted_hits = sorted(best_per_ref.values(), key=lambda r: r["matchCount"], reverse=True)[:top_k]
            for rank, hit in enumerate(sorted_hits, start=1):
                mid = int(hit["matchIdentifier"])
                rows_out.append({
                    "query_id": qid,
                    "rank": rank,
                    "predicted_ref_id": audio_id_map.get(mid),
                    "predicted_olaf_id": mid,
                    "match_count": int(hit["matchCount"]),
                    "ref_start": float(hit["referenceStart"]),
                    "ref_stop": float(hit["referenceStop"]),
                    "query_start": float(hit["queryStart"]),
                    "query_stop": float(hit["queryStop"]),
                    "latency_ms": lat,
                })
        except Exception as e:
            log.error(f"  QUERY FAILED for {qid}: {e!r}")
            rows_out.append({
                "query_id": qid, "rank": 0, "predicted_ref_id": None,
                "match_count": 0, "ref_start": None, "ref_stop": None,
                "query_start": None, "query_stop": None, "latency_ms": None,
            })

        if (i + 1) % 100 == 0:
            log.info(f"  queried {i+1}/{n}  (last latency {latencies_ms[-1]:.1f} ms)")

    df = pd.DataFrame(rows_out)
    stats = {
        "n_queries": n,
        "n_with_match": n_with_match,
        "n_no_match": n - n_with_match,
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
        "latency_ms_p99": float(np.percentile(latencies_ms, 99)) if latencies_ms else None,
        "latency_ms_max": float(np.max(latencies_ms)) if latencies_ms else None,
        "total_wall_sec": round(time.time() - t_outer, 2),
    }
    return df, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, type=Path)
    ap.add_argument("--queries", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-index", action="store_true",
                    help="skip indexing; reuse existing DB (loads audio_id_map from previous index_log.json)")
    args = ap.parse_args()

    out_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "olaf_runner.log", mode="w"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("olaf")

    log.info(f"refs:    {[str(p) for p in args.refs]}")
    log.info(f"queries: {[str(p) for p in args.queries]}")
    log.info(f"output:  {out_dir}")
    log.info(f"top_k:   {args.top_k}")

    refs_dfs = [pd.read_csv(p) for p in args.refs]
    refs_df = pd.concat(refs_dfs, ignore_index=True)
    log.info(f"loaded {len(refs_df)} refs across {len(args.refs)} files")

    queries_dfs = [pd.read_csv(p) for p in args.queries]
    queries_df = pd.concat(queries_dfs, ignore_index=True)
    log.info(f"loaded {len(queries_df)} queries across {len(args.queries)} files")

    if args.skip_index:
        idx_log_path = out_dir / "index_log.json"
        if not idx_log_path.exists():
            log.error("--skip-index requires a previous index_log.json in --output")
            return 2
        index_log = json.loads(idx_log_path.read_text())
        audio_id_map = {int(k): v for k, v in index_log["audio_id_map"].items()}
        log.info(f"reusing index of {len(audio_id_map)} refs (DB on disk: {OLAF_HOME_DB})")
    else:
        log.info("=== INDEX ===")
        index_log = index_refs(refs_df, log)
        # Don't write the full audio_id_map twice (it goes in the index_log too,
        # but make sure to retain it for skip-index runs).
        (out_dir / "index_log.json").write_text(json.dumps(index_log, indent=2))
        audio_id_map = {int(k): v for k, v in index_log["audio_id_map"].items()}
        log.info(f"INDEX done: {index_log['n_indexed']}/{index_log['n_refs']} refs, "
                 f"DB size {index_log['db_size_bytes']/1e6:.1f} MB, "
                 f"wall {index_log['total_wall_sec']:.1f}s")

    log.info("=== QUERY ===")
    results_df, query_stats = run_queries(queries_df, audio_id_map, args.top_k, log)
    results_df.to_parquet(out_dir / "query_results.parquet", index=False)
    (out_dir / "query_log.json").write_text(json.dumps(query_stats, indent=2))
    log.info(f"QUERY done: {query_stats['n_with_match']}/{query_stats['n_queries']} matched, "
             f"latency p50={query_stats['latency_ms_p50']:.1f}ms, "
             f"p95={query_stats['latency_ms_p95']:.1f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
