"""Deep monitor for the running Dejavu indexer.

Samples system memory + runner RSS + Postgres row counts every 30 s for the
duration given. Prints a table so we can see whether memory pressure is
trending up (abort risk) or stabilizing.
"""

from __future__ import annotations

import argparse
import time
import psutil
import psycopg2


def find_runner_rss_mb():
    for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "dejavu_runner.py" in cmd and "scripts/systems" in cmd:
                return p.info["memory_info"].rss / 1e6
        except Exception:
            continue
    return None


def db_counts():
    try:
        conn = psycopg2.connect(
            host="/Users/prita/.afp_pgserver",
            dbname="dejavu_saraga",
            user="postgres",
            connect_timeout=2,
        )
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM songs WHERE fingerprinted=1")
        n_songs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM fingerprints")
        n_fp = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n_songs, n_fp
    except Exception:
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8, help="number of samples")
    ap.add_argument("--interval", type=int, default=30, help="seconds between samples")
    args = ap.parse_args()

    header = "{:>4s} {:>5s} {:>9s} {:>6s} {:>11s} {:>8s}".format(
        "t_s", "mem%", "avail_GB", "songs", "fp_rows", "rss_MB"
    )
    print(header)
    t_start = time.time()
    for i in range(args.samples):
        t = int(time.time() - t_start)
        vm = psutil.virtual_memory()
        n_songs, n_fp = db_counts()
        rss = find_runner_rss_mb()
        print(
            "{:>4d} {:>5d} {:>9.2f} {:>6} {:>11} {:>8}".format(
                t,
                int(vm.percent),
                vm.available / 1e9,
                "?" if n_songs is None else str(n_songs),
                "?" if n_fp is None else f"{n_fp:,}",
                "-" if rss is None else f"{rss:.0f}",
            ),
            flush=True,
        )
        if i < args.samples - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
