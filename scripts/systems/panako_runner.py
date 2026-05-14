"""Panako system runner.

Mirrors olaf_runner.py / dejavu_runner.py:
  - Same output artifacts (index_log.json, query_results.parquet, query_log.json)
  - Same long-format result schema
  - Single-process; Panako runs in one JVM, queries batched
  - Uses STRATEGY=PANAKO (pitch/time-shift-invariant CQT triplet hashing),
    NOT Panako's bundled OLAF baseline strategy

Reproducibility:
  - Java 11 from project-local Temurin (external/tools/jdk-11.0.31+11)
  - ffmpeg + ffprobe arm64 from project-local (external/tools/ffmpeg)
  - Panako fat JAR built with shadow plugin (external/Panako/build/libs/panako-2.1-all.jar)
  - DB at $HOME/.panako/dbs/panako_db (Panako's tuned default)

Usage:
  uv run python scripts/systems/panako_runner.py \\
      --refs    data/manifests/hindustani/refs.csv \\
                data/manifests/carnatic/refs.csv \\
      --queries data/manifests/hindustani/queries.csv \\
                data/manifests/carnatic/queries.csv \\
      --output  data/results/panako/saraga_only_main \\
      --top-k   10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JDK_HOME = PROJECT_ROOT / "external" / "tools" / "jdk-11.0.31+11" / "Contents" / "Home"
FFMPEG_DIR = PROJECT_ROOT / "external" / "tools" / "ffmpeg"
PANAKO_JAR = PROJECT_ROOT / "external" / "Panako" / "build" / "libs" / "panako-2.1-all.jar"
PANAKO_HOME_DBS = Path.home() / ".panako" / "dbs"


def _env_for_panako() -> dict:
    env = os.environ.copy()
    env["JAVA_HOME"] = str(JDK_HOME)
    env["PATH"] = f"{JDK_HOME / 'bin'}:{FFMPEG_DIR}:{env.get('PATH', '')}"
    return env


def _java_jar(*args, env=None, capture_output=True) -> subprocess.CompletedProcess:
    cmd = [str(JDK_HOME / "bin" / "java"), "-Xmx4g", "-jar", str(PANAKO_JAR), *args]
    return subprocess.run(cmd, env=env or _env_for_panako(), capture_output=capture_output, text=True)


def _ensure_panako_strategy() -> None:
    """Verify config.properties next to the JAR has STRATEGY=PANAKO."""
    cfg = PANAKO_JAR.parent / "config.properties"
    if not cfg.exists():
        # First-run will create defaults — make Panako emit them by invoking config
        _java_jar("config", env=_env_for_panako(), capture_output=True)
    contents = cfg.read_text()
    new_lines = []
    set_strategy = False
    for line in contents.splitlines():
        if line.startswith("STRATEGY="):
            new_lines.append("STRATEGY=PANAKO")
            set_strategy = True
        else:
            new_lines.append(line)
    if not set_strategy:
        new_lines.append("STRATEGY=PANAKO")
    cfg.write_text("\n".join(new_lines) + "\n")


def clear_dbs() -> None:
    if PANAKO_HOME_DBS.exists():
        shutil.rmtree(PANAKO_HOME_DBS)
    PANAKO_HOME_DBS.mkdir(parents=True, exist_ok=True)


def db_size_bytes() -> int:
    if not PANAKO_HOME_DBS.exists():
        return 0
    return sum(p.stat().st_size for p in PANAKO_HOME_DBS.rglob("*") if p.is_file())


def index_refs(refs_df: pd.DataFrame, log: logging.Logger) -> dict:
    """Index all refs in ONE Panako JVM invocation via a list file."""
    clear_dbs()
    list_file = PROJECT_ROOT / "data" / "results" / "panako" / "_refs_list.txt"
    list_file.parent.mkdir(parents=True, exist_ok=True)

    # Build path list, capture path → ref_id map for later
    path_to_ref: dict[str, str] = {}
    with list_file.open("w") as f:
        for _, row in refs_df.iterrows():
            p = str(Path(row["audio_path"]).resolve())
            f.write(p + "\n")
            path_to_ref[p] = row["ref_id"]
    log.info(f"wrote ref list ({len(refs_df)} entries) to {list_file}")

    t0 = time.time()
    cmd = [str(JDK_HOME / "bin" / "java"), "-Xmx4g", "-jar", str(PANAKO_JAR), "store", str(list_file)]
    log.info(f"launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=_env_for_panako(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    n_stored = 0
    n_total_fp = 0
    fail_lines: list[str] = []
    panako_id_to_ref: dict[int, str] = {}
    for line in proc.stdout:
        line = line.rstrip()
        # Panako's per-track progress lines look like:
        # "INFO: 1; 357; <name>; <duration>; <elapsed s>; <speed>"
        # And "INFO: Stored <N> fingerprints for '<path>', id: <id>"
        if "INFO: Stored" in line and "fingerprints for" in line and ", id:" in line:
            # extract n, path, id
            try:
                # e.g. "INFO: Stored 12642 fingerprints for '<path>', id: 719393130"
                after_stored = line.split("Stored", 1)[1].strip()
                n_str = after_stored.split(" ", 1)[0]
                n_fp = int(n_str)
                rest = after_stored.split("'", 1)[1]
                path_quoted, after_path = rest.split("'", 1)
                pid = int(after_path.rsplit("id:", 1)[1].strip().rstrip("."))
                n_stored += 1
                n_total_fp += n_fp
                if path_quoted in path_to_ref:
                    panako_id_to_ref[pid] = path_to_ref[path_quoted]
                if n_stored % 25 == 0:
                    log.info(f"  indexed {n_stored}/{len(refs_df)}  (cumulative fingerprints={n_total_fp})")
            except Exception as e:
                fail_lines.append(f"parse fail: {line!r} ({e!r})")
        elif "WARNING" in line or ("ERROR" in line and "INFO" not in line):
            fail_lines.append(line[:300])
    proc.wait()
    elapsed = time.time() - t0
    return {
        "n_refs": len(refs_df),
        "n_indexed": n_stored,
        "n_failed": len(refs_df) - n_stored,
        "n_fingerprints_total": n_total_fp,
        "total_wall_sec": round(elapsed, 2),
        "db_size_bytes": int(db_size_bytes()),
        "fail_lines_head": fail_lines[:10],
        "exit_code": proc.returncode,
        "panako_id_to_ref": {str(k): v for k, v in panako_id_to_ref.items()},
    }


def parse_query_csv_line(line: str) -> dict | None:
    """Parse one Panako CSV match line.

    Verified from src/main/java/be/panako/cli/Panako.java:297
        String taskInfo = String.format("%d ; %d ; ", task, taskTotal);
    Fields:
      0  task         — running query index (1..N for N queries), NOT per-query rank
      1  taskTotal    — total number of queries in this run (constant per run)
      2  query path
      3  query start sec
      4  query stop sec
      5  match path
      6  panako match id (Jenkins hash)
      7  match start sec (in reference)
      8  match stop sec
      9  match score (aligned fingerprint count)
      10 time factor (1.0 = no time-stretch detected)
      11 frequency factor (1.0 = no pitch-shift)
      12 seconds with match (%)

    Per-query rank is implicit by emission order — first line for a given
    query is rank 1. The caller is responsible for assigning rank.
    """
    if ";" not in line:
        return None
    fields = [f.strip() for f in line.split(";")]
    if len(fields) < 13:
        return None
    if not fields[0].isdigit() or not fields[1].isdigit():
        return None
    try:
        return {
            "task": int(fields[0]),
            "task_total": int(fields[1]),
            "query_path": fields[2],
            "query_start_sec": float(fields[3]),
            "query_stop_sec": float(fields[4]),
            "ref_path": fields[5],
            "panako_match_id": int(fields[6]),
            "ref_start": float(fields[7]),
            "ref_stop": float(fields[8]),
            "match_count": int(fields[9]),
            "time_factor_pct": float(fields[10].rstrip(" %")),
            "freq_factor_pct": float(fields[11].rstrip(" %")),
            "seconds_with_match_pct": float(fields[12].rstrip(" %")),
        }
    except (ValueError, IndexError):
        return None


def run_queries(queries_df: pd.DataFrame, panako_id_to_ref: dict[int, str],
                top_k: int, log: logging.Logger) -> tuple[pd.DataFrame, dict]:
    list_file = PROJECT_ROOT / "data" / "results" / "panako" / "_queries_list.txt"
    list_file.parent.mkdir(parents=True, exist_ok=True)
    query_path_to_id: dict[str, str] = {}
    with list_file.open("w") as f:
        for _, row in queries_df.iterrows():
            p = str(Path(row["audio_path"]).resolve())
            f.write(p + "\n")
            query_path_to_id[p] = row["query_id"]
    log.info(f"wrote query list ({len(queries_df)} entries)")

    t0 = time.time()
    cmd = [
        str(JDK_HOME / "bin" / "java"),
        "-Xmx4g",
        f"-Dpanako.numberOfQueryResults={top_k}",  # informational; actual setting is in config.properties
        "-jar",
        str(PANAKO_JAR),
        "query",
        str(list_file),
    ]
    log.info(f"launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=_env_for_panako(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    rows: list[dict] = []
    # We capture per-query latency by watching for the "Query for N prints, M matches in T ms" INFO line
    cur_query_latency_ms: float | None = None
    seen_queries: set[str] = set()
    matches_by_query: dict[str, list[dict]] = {}

    for line in proc.stdout:
        line = line.rstrip()
        if "Query for" in line and " matches in " in line and "ms" in line:
            try:
                cur_query_latency_ms = float(line.rsplit("in ", 1)[1].split(" ", 1)[0])
            except Exception:
                cur_query_latency_ms = None
            continue
        parsed = parse_query_csv_line(line)
        if parsed is None:
            continue
        parsed["latency_ms"] = cur_query_latency_ms
        qid = query_path_to_id.get(parsed["query_path"])
        if qid is None:
            continue
        parsed["query_id"] = qid
        parsed["predicted_ref_id"] = panako_id_to_ref.get(parsed["panako_match_id"])
        # Rank within a query = emission order. We preserve insertion order
        # via list.append; rank assignment happens in the assembly step.
        matches_by_query.setdefault(qid, []).append(parsed)
        seen_queries.add(qid)

    proc.wait()
    elapsed = time.time() - t0

    # Assemble rows per query, ranks assigned 1..K in emission order, capped at top_k.
    for qid in queries_df["query_id"]:
        ms = matches_by_query.get(qid, [])
        if not ms:
            rows.append({
                "query_id": qid, "rank": 0, "predicted_ref_id": None,
                "panako_match_id": None, "match_count": 0,
                "ref_start": None, "ref_stop": None,
                "query_start": None, "query_stop": None,
                "time_factor_pct": None, "freq_factor_pct": None,
                "seconds_with_match_pct": None, "latency_ms": None,
            })
            continue
        # Panako emits best-first per query; preserve that order. Cap at top_k.
        for rank, hit in enumerate(ms[:top_k], start=1):
            rows.append({
                "query_id": qid,
                "rank": rank,
                "predicted_ref_id": hit.get("predicted_ref_id"),
                "panako_match_id": hit["panako_match_id"],
                "match_count": hit["match_count"],
                "ref_start": hit["ref_start"],
                "ref_stop": hit["ref_stop"],
                "query_start": hit["query_start_sec"],
                "query_stop": hit["query_stop_sec"],
                "time_factor_pct": hit["time_factor_pct"],
                "freq_factor_pct": hit["freq_factor_pct"],
                "seconds_with_match_pct": hit["seconds_with_match_pct"],
                "latency_ms": hit["latency_ms"],
            })

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    df = pd.DataFrame(rows)
    stats = {
        "n_queries": len(queries_df),
        "n_with_match": len(seen_queries),
        "n_no_match": len(queries_df) - len(seen_queries),
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else None,
        "latency_ms_p99": float(np.percentile(latencies, 99)) if latencies else None,
        "latency_ms_max": float(np.max(latencies)) if latencies else None,
        "total_wall_sec": round(elapsed, 2),
        "exit_code": proc.returncode,
    }
    return df, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, type=Path)
    ap.add_argument("--queries", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    out_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "panako_runner.log", mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("panako")
    log.info(f"refs:    {[str(p) for p in args.refs]}")
    log.info(f"queries: {[str(p) for p in args.queries]}")
    log.info(f"output:  {out_dir}")
    log.info(f"top_k:   {args.top_k}")
    log.info(f"PANAKO_JAR: {PANAKO_JAR}")
    log.info(f"JDK_HOME:   {JDK_HOME}")

    if not PANAKO_JAR.exists():
        log.error(f"Panako fat JAR missing: {PANAKO_JAR}")
        return 2

    _ensure_panako_strategy()
    log.info("strategy set to PANAKO in config.properties")

    refs_df = pd.concat([pd.read_csv(p) for p in args.refs], ignore_index=True)
    queries_df = pd.concat([pd.read_csv(p) for p in args.queries], ignore_index=True)
    log.info(f"loaded {len(refs_df)} refs and {len(queries_df)} queries")

    if args.skip_index:
        idx_log_path = out_dir / "index_log.json"
        if not idx_log_path.exists():
            log.error("--skip-index requires a previous index_log.json in --output")
            return 3
        index_log = json.loads(idx_log_path.read_text())
        panako_id_to_ref = {int(k): v for k, v in index_log["panako_id_to_ref"].items()}
        log.info(f"reusing index of {len(panako_id_to_ref)} refs")
    else:
        log.info("=== INDEX ===")
        index_log = index_refs(refs_df, log)
        (out_dir / "index_log.json").write_text(json.dumps(index_log, indent=2))
        panako_id_to_ref = {int(k): v for k, v in index_log["panako_id_to_ref"].items()}
        log.info(f"INDEX done: {index_log['n_indexed']}/{index_log['n_refs']} refs, "
                 f"fingerprints={index_log['n_fingerprints_total']}, "
                 f"db_size={index_log['db_size_bytes']/1e6:.0f} MB, "
                 f"wall={index_log['total_wall_sec']:.0f}s")

    log.info("=== QUERY ===")
    df, stats = run_queries(queries_df, panako_id_to_ref, args.top_k, log)
    df.to_parquet(out_dir / "query_results.parquet", index=False)
    (out_dir / "query_log.json").write_text(json.dumps(stats, indent=2))
    log.info(f"QUERY done: {stats['n_with_match']}/{stats['n_queries']} matched, "
             f"latency p50={stats['latency_ms_p50']:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
