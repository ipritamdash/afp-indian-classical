"""Download a Saraga sub-corpus via mirdata into a project-local data_home.

Usage:
    uv run python scripts/download_saraga.py carnatic
    uv run python scripts/download_saraga.py hindustani

The script prints periodic progress so background log tailing is informative.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mirdata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME_ROOT = PROJECT_ROOT / "data" / "mirdata"


def download(corpus: str) -> int:
    if corpus not in ("carnatic", "hindustani"):
        print(f"ERROR: corpus must be carnatic or hindustani, got {corpus!r}")
        return 2

    dataset_name = f"saraga_{corpus}"
    data_home = DATA_HOME_ROOT / dataset_name
    data_home.mkdir(parents=True, exist_ok=True)

    print(f"[{corpus}] data_home: {data_home}", flush=True)
    print(f"[{corpus}] initializing mirdata dataset {dataset_name!r}", flush=True)
    ds = mirdata.initialize(dataset_name, data_home=str(data_home))

    print(f"[{corpus}] remotes:", flush=True)
    for k, r in ds.remotes.items():
        print(f"  [{corpus}]   {k}: {r.url}  -> {r.filename}", flush=True)

    t0 = time.time()
    print(f"[{corpus}] starting download.download() (zip + extract)…", flush=True)
    ds.download(cleanup=False)  # keep the zip for later auditing
    dt = time.time() - t0
    print(f"[{corpus}] download finished in {dt:.0f}s", flush=True)

    print(f"[{corpus}] validating index (this also confirms files unpacked correctly)…", flush=True)
    t0 = time.time()
    missing_files, invalid_checksums = ds.validate(verbose=False)
    dt = time.time() - t0
    print(
        f"[{corpus}] validation: missing_files={sum(len(v) for v in missing_files.values())} "
        f"invalid_checksums={sum(len(v) for v in invalid_checksums.values())} ({dt:.0f}s)",
        flush=True,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["carnatic", "hindustani"])
    args = p.parse_args()
    return download(args.corpus)


if __name__ == "__main__":
    sys.exit(main())
