"""Download every file in a Zenodo record, with resume support + checksum verify.

Usage:
    uv run python scripts/download_zenodo.py 6584259 data/zenodo/semi_classical
    uv run python scripts/download_zenodo.py 6584021 data/zenodo/folk
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def md5_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024:
            return f"{f:.1f}{unit}"
        f /= 1024
    return f"{f:.1f}PB"


def fetch_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def download_one(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Resumable download with range requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    have = tmp.stat().st_size if tmp.exists() else 0
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  [skip] {dest.name} already complete", flush=True)
        return
    headers = {"User-Agent": "afp-bench/0.1"}
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        print(f"  [resume] {dest.name} from {fmt_bytes(have)}", flush=True)
    else:
        print(f"  [start] {dest.name}", flush=True)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("ab") as out:
        last_log = time.time()
        downloaded = have
        chunk = 1024 * 256
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            downloaded += len(buf)
            now = time.time()
            if now - last_log > 10:
                if expected_size:
                    pct = 100 * downloaded / expected_size
                    print(f"  [{dest.name}] {fmt_bytes(downloaded)}/{fmt_bytes(expected_size)} ({pct:.0f}%)", flush=True)
                else:
                    print(f"  [{dest.name}] {fmt_bytes(downloaded)}", flush=True)
                last_log = now
    tmp.rename(dest)
    print(f"  [done] {dest.name} {fmt_bytes(dest.stat().st_size)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--verify-md5", action="store_true", help="verify md5 after download (slow)")
    ap.add_argument(
        "--include",
        action="append",
        default=None,
        help="only download files whose key contains this substring (repeatable). default: all files",
    )
    args = ap.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[zenodo:{args.record_id}] fetching record metadata…", flush=True)
    rec = fetch_record(args.record_id)
    title = rec.get("metadata", {}).get("title", "")
    license_id = rec.get("metadata", {}).get("license", {}).get("id", "")
    files = rec.get("files", [])
    if args.include:
        before = len(files)
        files = [f for f in files if any(inc in f["key"] for inc in args.include)]
        print(f"[zenodo:{args.record_id}] include filter: {args.include} → {len(files)}/{before} files", flush=True)
    total = sum(f["size"] for f in files)
    print(f"[zenodo:{args.record_id}] {title}", flush=True)
    print(f"[zenodo:{args.record_id}] license: {license_id}", flush=True)
    print(f"[zenodo:{args.record_id}] {len(files)} files, total {fmt_bytes(total)}", flush=True)

    # Persist the manifest so downstream code (extraction, inspection) doesn't re-hit the API
    manifest = {
        "record_id": args.record_id,
        "title": title,
        "license": license_id,
        "files": [{"key": f["key"], "size": f["size"], "checksum": f.get("checksum"), "url": f["links"]["self"]} for f in files],
    }
    (out_dir / "_zenodo_manifest.json").write_text(json.dumps(manifest, indent=2))

    for f in files:
        dest = out_dir / f["key"]
        download_one(f["links"]["self"], dest, expected_size=f["size"])

    if args.verify_md5:
        print(f"[zenodo:{args.record_id}] verifying md5…", flush=True)
        for f in files:
            dest = out_dir / f["key"]
            cs = (f.get("checksum") or "").removeprefix("md5:")
            if not cs:
                print(f"  [skip-md5] {f['key']} (no checksum in record)")
                continue
            got = md5_of_file(dest)
            ok = got == cs
            print(f"  [md5 {'OK' if ok else 'FAIL'}] {f['key']}")
            if not ok:
                return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
