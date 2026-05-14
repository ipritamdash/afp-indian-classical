"""Download FMA (Free Music Archive) zips from the Swiss mirror.

Usage:
    uv run python scripts/download_fma.py small data/fma
    uv run python scripts/download_fma.py medium data/fma
    uv run python scripts/download_fma.py large data/fma
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FMA_URLS = {
    # Sizes verified against server Content-Length on 2026-05-11. If a future
    # FMA release changes the zip, re-HEAD and update.
    "metadata": ("https://os.unil.cloud.switch.ch/fma/fma_metadata.zip", 358_553_516),
    "small": ("https://os.unil.cloud.switch.ch/fma/fma_small.zip", 7_695_413_172),
    "medium": ("https://os.unil.cloud.switch.ch/fma/fma_medium.zip", 23_825_005_356),
    "large": ("https://os.unil.cloud.switch.ch/fma/fma_large.zip", 99_955_834_888),
}


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def download_resumable(url: str, dest: Path, expected_size: int, max_retries: int = 20) -> None:
    """Resumable download with retry on premature server-side disconnect.

    Guarantees: the destination file is only renamed from .part → .zip when its
    size exactly matches expected_size. Premature closes loop until success
    or max_retries exhausted (with exponential backoff on consecutive failures).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} complete", flush=True)
        return

    attempt = 0
    consecutive_no_progress = 0
    last_have = -1
    while True:
        have = tmp.stat().st_size if tmp.exists() else 0
        if have >= expected_size:
            break

        if have == last_have:
            consecutive_no_progress += 1
            if consecutive_no_progress > max_retries:
                raise RuntimeError(f"[{dest.name}] stuck at {fmt_bytes(have)} after {max_retries} retries without progress")
            wait = min(60, 2 ** min(consecutive_no_progress, 6))
            print(f"[wait] no progress on {dest.name}, sleeping {wait}s before retry", flush=True)
            time.sleep(wait)
        else:
            consecutive_no_progress = 0
        last_have = have
        attempt += 1

        headers = {"User-Agent": "afp-bench/0.1"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
            print(f"[resume #{attempt}] {dest.name} from {fmt_bytes(have)}/{fmt_bytes(expected_size)} ({100*have/expected_size:.1f}%)", flush=True)
        else:
            print(f"[start] {dest.name} (expect {fmt_bytes(expected_size)})", flush=True)

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("ab") as out:
                downloaded = have
                last_log = time.time()
                while True:
                    buf = resp.read(1024 * 256)
                    if not buf:
                        break
                    out.write(buf)
                    downloaded += len(buf)
                    now = time.time()
                    if now - last_log > 10:
                        pct = 100 * downloaded / expected_size
                        print(f"[{dest.name}] {fmt_bytes(downloaded)}/{fmt_bytes(expected_size)} ({pct:.1f}%)", flush=True)
                        last_log = now
        except Exception as e:
            print(f"[error] {dest.name}: {e!r}; will retry with Range", flush=True)
            continue

    # post-validate before renaming
    actual = tmp.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"[{dest.name}] size mismatch after download: got {actual}, expected {expected_size}")
    tmp.rename(dest)
    print(f"[done] {dest.name} {fmt_bytes(dest.stat().st_size)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(FMA_URLS.keys()))
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    out_dir = (PROJECT_ROOT / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    url, sz = FMA_URLS[args.variant]
    dest = out_dir / f"fma_{args.variant}.zip"
    download_resumable(url, dest, sz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
