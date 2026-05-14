"""Compute md5 for any reference track that doesn't have one yet, in-place.

After running build_testset.py, only the 100 test-source tracks per corpus have
source_md5 populated (because md5 was computed lazily inside the query-cut
loop). For full reproducibility of the library state, hash every ref.

Safe to re-run — only fills missing md5s.

Usage:
    uv run python scripts/audit_refs.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = PROJECT_ROOT / "data" / "manifests"


def md5_of(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fmt_time(s: float) -> str:
    return f"{int(s//60)}m{int(s%60):02d}s"


def main() -> int:
    for corpus in ("hindustani", "carnatic"):
        refs_csv = MANIFESTS / corpus / "refs.csv"
        if not refs_csv.exists():
            print(f"[{corpus}] no refs.csv, skipping")
            continue
        df = pd.read_csv(refs_csv)
        if "source_md5" not in df.columns:
            df["source_md5"] = ""
        df["source_md5"] = df["source_md5"].fillna("").astype(str)

        need = df[df["source_md5"].str.len() < 32]
        already = len(df) - len(need)
        print(f"[{corpus}] {len(df)} refs, {already} already hashed, {len(need)} to hash")

        t0 = time.time()
        total_bytes = 0
        for idx, row in need.iterrows():
            p = Path(row["audio_path"])
            if not p.exists():
                print(f"  [{corpus}] missing audio: {row['source_track_id']} -> {p}")
                continue
            digest = md5_of(p)
            df.at[idx, "source_md5"] = digest
            total_bytes += p.stat().st_size
            if (need.index.get_loc(idx) + 1) % 25 == 0:
                rate_mb = total_bytes / 1e6 / max(1e-3, time.time() - t0)
                print(f"  [{corpus}] hashed {need.index.get_loc(idx)+1}/{len(need)} ({rate_mb:.0f} MB/s)")

        df.to_csv(refs_csv, index=False)
        print(f"[{corpus}] WROTE {refs_csv} (md5 populated for all rows) in {fmt_time(time.time()-t0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
