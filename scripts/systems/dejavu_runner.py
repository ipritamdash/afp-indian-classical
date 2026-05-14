"""Dejavu system runner using pgserver-managed Postgres.

Mirrors olaf_runner.py:
  * Same output artifacts (index_log.json, query_results.parquet, query_log.json)
  * Same long-format result rows (query_id, rank, predicted_ref_id, score, ref_start, ref_stop, latency_ms)
  * Sequential execution (single-threaded; Dejavu is not designed for in-process parallelism)
  * Bypasses pydub/ffmpeg — audio loaded directly via soundfile + librosa.resample

Reproducibility:
  * Postgres data dir: $HOME/.afp_pgserver (no spaces; reproducible from clone)
  * Database name: dejavu_saraga (dropped + recreated on each --index run)
  * Sample rate: 44100 Hz (Dejavu's tuned constants assume this — confirmed by reading
    src/dejavu/__init__.py align_matches which converts frame→sec using DEFAULT_FS=44100)

Usage:
  uv run python scripts/systems/dejavu_runner.py \\
      --refs    data/manifests/hindustani/refs.csv \\
                data/manifests/carnatic/refs.csv \\
      --queries data/manifests/hindustani/queries_44k.csv \\
                data/manifests/carnatic/queries_44k.csv \\
      --output  data/results/dejavu/saraga_only_main \\
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
import psycopg2
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEJAVU_DIR = PROJECT_ROOT / "external" / "dejavu"
sys.path.insert(0, str(DEJAVU_DIR))

from dejavu import Dejavu  # noqa: E402
from dejavu.config import settings as dejavu_settings  # noqa: E402
from dejavu.config.settings import (  # noqa: E402
    DEFAULT_FS, FIELD_HASH, FIELD_OFFSET, FIELD_SONG_ID,
    FINGERPRINTS_TABLENAME,
    HASHES_MATCHED, INPUT_CONFIDENCE, FINGERPRINTED_CONFIDENCE,
    OFFSET, OFFSET_SECS, SONG_ID, SONG_NAME,
)
from dejavu.database_handler.postgres_database import PostgreSQLDatabase  # noqa: E402
from psycopg2.extras import execute_values  # noqa: E402


class FastPostgreSQLDatabase(PostgreSQLDatabase):
    """Bulk-insert override.

    Dejavu's default insert_hashes uses cur.executemany() which sends one
    INSERT statement per row. For Saraga's ~230k hashes/track this becomes
    the dominant cost. psycopg2.extras.execute_values batches into a single
    INSERT VALUES (..),(..),..,(..) statement per page — 5-10× faster, same
    ON CONFLICT DO NOTHING semantics, same final database state.
    """

    def insert_hashes(self, song_id: int, hashes, batch_size: int = 2000) -> None:
        if not hashes:
            return
        values = [(song_id, hsh, int(offset)) for hsh, offset in hashes]
        sql = (
            f'INSERT INTO "{FINGERPRINTS_TABLENAME}" '
            f'("{FIELD_SONG_ID}", "{FIELD_HASH}", "{FIELD_OFFSET}") '
            f'VALUES %s ON CONFLICT DO NOTHING'
        )
        template = "(%s, decode(%s, 'hex'), %s)"
        with self.cursor() as cur:
            execute_values(cur, sql, values, template=template, page_size=batch_size)


# Replace Dejavu's postgres backend with our bulk-insert subclass.
#
# We must patch BOTH `dejavu.base_classes.base_database.get_database` AND
# the reference that Dejavu's package __init__ imported into its own
# namespace at startup: `dejavu/__init__.py` does
#   `from dejavu.base_classes.base_database import get_database`
# which captures the original function pointer. Calls inside Dejavu.__init__
# resolve `get_database` via the dejavu-module namespace, so patching only
# the source module would leave dejavu's local reference stale.
import dejavu as _dejavu_pkg  # noqa: E402
from dejavu.base_classes import base_database as _base_db_module  # noqa: E402

_original_get_database = _base_db_module.get_database


def _patched_get_database(db_type: str = "mysql"):
    if db_type.lower() == "postgres":
        return FastPostgreSQLDatabase
    return _original_get_database(db_type)


_base_db_module.get_database = _patched_get_database
_dejavu_pkg.get_database = _patched_get_database


PG_DATA_DIR = Path.home() / ".afp_pgserver"
DB_NAME = "dejavu_saraga"
DJV_FS = DEFAULT_FS  # 44100; locked by Dejavu's align_matches conversion


def ensure_pgserver():
    import pgserver
    PG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return pgserver.get_server(str(PG_DATA_DIR), cleanup_mode=None)


def recreate_db():
    """Drop and recreate the dejavu_saraga database; tables are created by Dejavu().setup()."""
    conn = psycopg2.connect(host=str(PG_DATA_DIR), dbname="postgres", user="postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
    cur.execute(f"CREATE DATABASE {DB_NAME}")
    cur.close()
    conn.close()


def db_size_bytes() -> int:
    """Total size of the Postgres database files on disk."""
    if not PG_DATA_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in PG_DATA_DIR.rglob("*") if p.is_file())


def build_dejavu(connect_db: str = DB_NAME) -> Dejavu:
    """Construct a Dejavu instance pointing at our pgserver Postgres."""
    config = {
        "database_type": "postgres",
        "database": {
            "host": str(PG_DATA_DIR),  # Unix socket dir (no spaces)
            "user": "postgres",
            "dbname": connect_db,
        },
    }
    return Dejavu(config)


def load_audio_44k(path: str) -> np.ndarray:
    """Mono int16 audio at 44.1 kHz. Dejavu's fingerprint() applies a fixed
    AMP_MIN=10 (dB) threshold tuned for int16-scale spectrogram magnitudes —
    its own decoder.read produces int16 via pydub. Passing float32 normalized
    to ±1 yields a spectrogram in dB ~ −50, all below threshold, giving zero
    fingerprints. We therefore scale to int16 to match Dejavu's contract.
    """
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        data = f.read(dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != DJV_FS:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=DJV_FS, res_type="soxr_hq")
    # Float32 audio is normalized to ±1 by soundfile. Scale to int16 range.
    return np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16)


def _fp_worker(payload):
    """Spawn-safe worker: loads audio + extracts Dejavu fingerprints.

    Returns (ref_id, file_hash, list_of_hashes, fp_sec_in_worker, peak_rss_mb).
    All imports done inside the function — macOS Pool uses spawn, no module
    state is inherited reliably.

    Memory discipline: aggressively drops the audio + spectrogram buffers
    before returning so the worker's RSS resets to baseline before the next
    task. Otherwise spectrograms from long Saraga tracks (~1.5 GB) accumulate.
    """
    import gc
    import os
    import sys
    import time
    import warnings
    from pathlib import Path
    warnings.filterwarnings("ignore")  # silence pydub's ffmpeg-not-found warning

    ref_id, audio_path, file_hash = payload
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    sys.path.insert(0, str(project_root / "external" / "dejavu"))

    import numpy as np
    import soundfile as sf
    import librosa
    import psutil
    from dejavu.logic.fingerprint import fingerprint as fp_fn

    t0 = time.time()
    try:
        with sf.SoundFile(audio_path) as f:
            sr = f.samplerate
            data = f.read(dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        del data  # free stereo buffer ASAP
        if sr != 44100:
            mono_r = librosa.resample(mono, orig_sr=sr, target_sr=44100, res_type="soxr_hq")
            del mono
            mono = mono_r
        samples = np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16)
        del mono  # free float32 buffer; only int16 remains

        hashes = list(fp_fn(samples, Fs=44100))
        del samples
    finally:
        peak_rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        gc.collect()

    return ref_id, file_hash, hashes, round(time.time() - t0, 2), round(peak_rss_mb, 1)


def _preflight_memory_check(min_available_gb: float = 8.0) -> tuple[bool, dict]:
    """Senior gate: refuse to launch parallel workers without RAM headroom.
    Returns (ok, info_dict). info_dict has total_gb, available_gb, percent_used."""
    import psutil
    vm = psutil.virtual_memory()
    info = {
        "total_gb": round(vm.total / 1e9, 2),
        "available_gb": round(vm.available / 1e9, 2),
        "percent_used": int(vm.percent),
        "threshold_gb": min_available_gb,
    }
    return vm.available >= min_available_gb * 1e9, info


def index_refs(djv: Dejavu, refs_df: pd.DataFrame, log: logging.Logger,
               n_workers: int = 1, abort_at_used_pct: int = 88) -> dict:
    """Index a reference library with bounded parallelism + memory monitoring.

    Safety contract:
      - Pre-flight RAM gate runs in main() before this is called.
      - Refs are sorted by duration descending → the longest tracks (largest
        per-worker RAM peak) hit the pool FIRST, so peak memory pressure is
        front-loaded and predictable. After the first ~n_workers tasks, all
        subsequent tasks are smaller and peak only decreases.
      - After every task completion the main process reads system memory
        percent. If it crosses `abort_at_used_pct`, the pool is shut down
        cleanly and partial progress is returned + persisted.
    """
    import psutil

    failed: list[tuple[str, str]] = []
    insert_sec = 0.0
    fp_sec_worker = 0.0
    songid_to_refid: dict[int, str] = {}
    n_hashes_total = 0
    peak_worker_rss_mb = 0.0
    peak_system_used_pct = 0
    aborted = False
    t_outer = time.time()

    # Pre-validate md5 to fail fast before launching workers
    refs_sorted = refs_df.sort_values("duration_sec", ascending=False).reset_index(drop=True)
    payloads = []
    for _, row in refs_sorted.iterrows():
        fh = (row.get("source_md5") or "").strip()
        if len(fh) != 32:
            failed.append((row["ref_id"], "missing source_md5 in refs.csv"))
            log.error(f"  {row['ref_id']}: missing source_md5; skipping")
            continue
        payloads.append((row["ref_id"], row["audio_path"], fh))

    log.info(f"index_refs: {len(payloads)} valid payloads, n_workers={n_workers}, abort_at_used_pct={abort_at_used_pct}")
    log.info(f"  top-5 longest tracks (will be processed FIRST): {[f'{p[0]}({d:.0f}s)' for p, d in list(zip(payloads, refs_sorted['duration_sec']))[:5]]}")
    n_done = 0

    def _record_insert(ref_id, file_hash, hashes, fp_t, w_rss):
        """Mutate closure state for one successful task; returns (failed_record_or_None)."""
        nonlocal insert_sec, fp_sec_worker, n_hashes_total, peak_worker_rss_mb
        fp_sec_worker += fp_t
        peak_worker_rss_mb = max(peak_worker_rss_mb, w_rss)
        try:
            t0 = time.time()
            sid = djv.db.insert_song(ref_id, file_hash, len(hashes))
            djv.db.insert_hashes(sid, hashes)
            djv.db.set_song_fingerprinted(sid)
            insert_sec += time.time() - t0
            songid_to_refid[int(sid)] = ref_id
            n_hashes_total += len(hashes)
            return None
        except Exception as e:
            return (ref_id, repr(e))

    if n_workers <= 1:
        for payload in payloads:
            try:
                ref_id, file_hash, hashes, fp_t, w_rss = _fp_worker(payload)
                rec = _record_insert(ref_id, file_hash, hashes, fp_t, w_rss)
                if rec is not None:
                    failed.append(rec)
                    log.error(f"  INSERT failed for {rec[0]}: {rec[1]}")
            except Exception as e:
                failed.append((payload[0], repr(e)))
                log.error(f"  STORE failed for {payload[0]}: {e!r}")
            n_done += 1
            sys_pct = psutil.virtual_memory().percent
            peak_system_used_pct = max(peak_system_used_pct, sys_pct)
            if sys_pct >= abort_at_used_pct:
                log.error(f"  ABORTING: system RAM used {sys_pct}% ≥ threshold {abort_at_used_pct}% after {n_done} tracks")
                aborted = True
                break
            if n_done % 25 == 0:
                log.info(f"  indexed {n_done}/{len(payloads)}  fp_sum={fp_sec_worker:.0f}s insert={insert_sec:.0f}s wall={time.time()-t_outer:.0f}s sys_mem={sys_pct}% peak_worker_rss={peak_worker_rss_mb:.0f}MB hashes={n_hashes_total}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            iterator = pool.imap_unordered(_fp_worker, payloads, chunksize=1)
            try:
                for ref_id, file_hash, hashes, fp_t, w_rss in iterator:
                    rec = _record_insert(ref_id, file_hash, hashes, fp_t, w_rss)
                    if rec is not None:
                        failed.append(rec)
                        log.error(f"  INSERT failed for {rec[0]}: {rec[1]}")
                    n_done += 1
                    sys_pct = psutil.virtual_memory().percent
                    peak_system_used_pct = max(peak_system_used_pct, sys_pct)
                    if sys_pct >= abort_at_used_pct:
                        log.error(f"  ABORTING: system RAM used {sys_pct}% ≥ threshold {abort_at_used_pct}% after {n_done} tracks (peak_worker_rss={peak_worker_rss_mb:.0f}MB)")
                        aborted = True
                        pool.terminate()
                        pool.join()
                        break
                    if n_done % 25 == 0:
                        log.info(f"  indexed {n_done}/{len(payloads)}  fp_sum={fp_sec_worker:.0f}s insert={insert_sec:.0f}s wall={time.time()-t_outer:.0f}s sys_mem={sys_pct}% peak_worker_rss={peak_worker_rss_mb:.0f}MB hashes={n_hashes_total}")
            except Exception as e:
                log.error(f"  pool iteration error: {e!r}")
                pool.terminate()
                pool.join()
                aborted = True

    return {
        "n_refs": len(refs_df),
        "n_indexed": len(refs_df) - len(failed),
        "n_failed": len(failed),
        "failed_samples": failed[:10],
        "worker_fp_sec_sum": round(fp_sec_worker, 2),
        "insert_sec": round(insert_sec, 2),
        "total_wall_sec": round(time.time() - t_outer, 2),
        "n_hashes_total": int(n_hashes_total),
        "db_size_bytes": int(db_size_bytes()),
        "n_workers": n_workers,
        "peak_worker_rss_mb": round(peak_worker_rss_mb, 1),
        "peak_system_used_pct": int(peak_system_used_pct),
        "aborted": aborted,
        "songid_to_refid": {str(k): v for k, v in songid_to_refid.items()},
    }


def run_queries(djv: Dejavu, queries_df: pd.DataFrame, songid_to_refid: dict[int, str],
                top_k: int, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    latencies_ms: list[float] = []
    n_with_match = 0
    n = len(queries_df)
    t_outer = time.time()

    for i, (_, row) in enumerate(queries_df.iterrows()):
        qid = row["query_id"]
        qpath = row["audio_path"]
        try:
            y = load_audio_44k(qpath)
            t0 = time.time()
            hashes, _fp_t = djv.generate_fingerprints(y, Fs=DJV_FS)
            matches, dedup_hashes, _q_t = djv.find_matches(hashes)
            aligned = djv.align_matches(matches, dedup_hashes, len(hashes), topn=top_k)
            lat = (time.time() - t0) * 1000.0
            latencies_ms.append(lat)

            if not aligned:
                rows.append({
                    "query_id": qid, "rank": 0, "predicted_ref_id": None,
                    "predicted_song_id": None, "match_count": 0,
                    "ref_start": None, "ref_stop": None,
                    "query_start": None, "query_stop": None,
                    "latency_ms": lat,
                })
                continue

            n_with_match += 1
            for rank, hit in enumerate(aligned, start=1):
                sid = int(hit[SONG_ID])
                rows.append({
                    "query_id": qid,
                    "rank": rank,
                    "predicted_ref_id": songid_to_refid.get(sid),
                    "predicted_song_id": sid,
                    "match_count": int(hit[HASHES_MATCHED]),
                    "ref_start": float(hit[OFFSET_SECS]),
                    # FIXED: use actual query length, not hardcoded 10.0 s.
                    # Bug: previous version emitted +10.0 regardless of length,
                    # silently wrong on 1/3/5 s splits. Does not affect HR/MRR/
                    # align_err (those don't use ref_stop), but the parquet column
                    # itself is now correct for downstream consumers.
                    "ref_stop": float(hit[OFFSET_SECS]) + float(row["length_sec"]),
                    "query_start": 0.0,  # Dejavu reports anchor-of-query, no inner offset
                    "query_stop": float(row["length_sec"]),
                    "latency_ms": lat,
                    "input_confidence": float(hit[INPUT_CONFIDENCE]),
                    "fingerprinted_confidence": float(hit[FINGERPRINTED_CONFIDENCE]),
                })
        except Exception as e:
            log.error(f"  QUERY failed for {qid}: {e!r}")
            rows.append({
                "query_id": qid, "rank": 0, "predicted_ref_id": None,
                "predicted_song_id": None, "match_count": 0,
                "ref_start": None, "ref_stop": None,
                "query_start": None, "query_stop": None,
                "latency_ms": None,
            })

        if (i + 1) % 100 == 0:
            log.info(f"  queried {i+1}/{n}  (last latency {latencies_ms[-1]:.1f} ms)")

    df = pd.DataFrame(rows)
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
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel fingerprint-extraction workers (single-writer Postgres insert)")
    ap.add_argument("--skip-index", action="store_true",
                    help="reuse existing db; loads songid_to_refid from previous index_log.json")
    args = ap.parse_args()

    out_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "dejavu_runner.log", mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("dejavu")
    log.info(f"refs:    {[str(p) for p in args.refs]}")
    log.info(f"queries: {[str(p) for p in args.queries]}")
    log.info(f"output:  {out_dir}")
    log.info(f"top_k:   {args.top_k}")

    # Pre-flight RAM gate — required by the senior safety contract. See memory
    # entry feedback_audio_parallelism.md.
    min_available_gb = 8.0 if args.workers >= 2 else 3.0
    ok, info = _preflight_memory_check(min_available_gb=min_available_gb)
    log.info(f"pre-flight: total={info['total_gb']} GB, available={info['available_gb']} GB, used={info['percent_used']}%, required≥{info['threshold_gb']} GB for workers={args.workers}")
    if not ok:
        log.error(f"PRE-FLIGHT FAILED. Free RAM until available ≥ {info['threshold_gb']} GB, or reduce --workers.")
        return 4

    log.info("starting pgserver…")
    srv = ensure_pgserver()
    log.info(f"pgserver URI: {srv.get_uri()}")

    refs_df = pd.concat([pd.read_csv(p) for p in args.refs], ignore_index=True)
    queries_df = pd.concat([pd.read_csv(p) for p in args.queries], ignore_index=True)
    log.info(f"loaded {len(refs_df)} refs and {len(queries_df)} queries")

    if args.skip_index:
        idx_log_path = out_dir / "index_log.json"
        if not idx_log_path.exists():
            log.error("--skip-index requires a previous index_log.json in --output")
            return 2
        index_log = json.loads(idx_log_path.read_text())
        songid_to_refid = {int(k): v for k, v in index_log["songid_to_refid"].items()}
        log.info(f"reusing index of {len(songid_to_refid)} songs (DB at {PG_DATA_DIR}/{DB_NAME})")
        djv = build_dejavu()
    else:
        log.info(f"recreating database {DB_NAME}")
        recreate_db()
        djv = build_dejavu()  # also runs setup() → creates tables
        log.info("=== INDEX ===")
        index_log = index_refs(djv, refs_df, log, n_workers=args.workers)
        (out_dir / "index_log.json").write_text(json.dumps(index_log, indent=2))
        songid_to_refid = {int(k): v for k, v in index_log["songid_to_refid"].items()}
        log.info(f"INDEX done: {index_log['n_indexed']}/{index_log['n_refs']} refs  "
                 f"hashes={index_log['n_hashes_total']}  "
                 f"db_size={index_log['db_size_bytes']/1e6:.1f} MB  "
                 f"wall={index_log['total_wall_sec']:.1f}s")

    log.info("=== QUERY ===")
    results_df, query_stats = run_queries(djv, queries_df, songid_to_refid, args.top_k, log)
    results_df.to_parquet(out_dir / "query_results.parquet", index=False)
    (out_dir / "query_log.json").write_text(json.dumps(query_stats, indent=2))
    log.info(f"QUERY done: {query_stats['n_with_match']}/{query_stats['n_queries']} matched  "
             f"latency p50={query_stats['latency_ms_p50']:.1f}ms  p95={query_stats['latency_ms_p95']:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
