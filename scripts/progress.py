"""Quick non-spammy status of background downloads.

Usage:
    uv run python scripts/progress.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"


EXPECTED_BYTES = {
    "saraga_hindustani": 4_109_172_493,
    "saraga_carnatic": 14_376_627_433,
    # Indian Regional Music (Zenodo 5825830) and Indian Semi-Classical (Zenodo 6584259)
    # sizes will be added at download time
}


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}PB"


def main() -> None:
    # Active python download processes
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid,etime,command"],
            capture_output=True, text=True, check=False,
        ).stdout
    except Exception:
        ps = ""
    download_lines = [
        ln for ln in ps.splitlines()
        if any(needle in ln for needle in ("download_saraga", "download_fma", "download_zenodo", "zenodo_get", "zenodo-get"))
    ]
    print("=== active download processes ===")
    if download_lines:
        for ln in download_lines:
            print("  " + ln.strip())
    else:
        print("  (none)")

    print()
    print("=== dataset directory sizes ===")
    for sub in sorted((DATA / "mirdata").iterdir() if (DATA / "mirdata").exists() else []):
        if sub.is_dir():
            size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
            expected = EXPECTED_BYTES.get(sub.name)
            pct = f" ({100*size/expected:.0f}% of {fmt_bytes(expected)})" if expected else ""
            print(f"  {sub.name}: {fmt_bytes(size)}{pct}")
    for sub in sorted((DATA / "zenodo").iterdir() if (DATA / "zenodo").exists() else []):
        if sub.is_dir():
            size = sum(p.stat().st_size for p in sub.rglob("*") if p.is_file())
            print(f"  zenodo/{sub.name}: {fmt_bytes(size)}")
    if (DATA / "fma").exists():
        for p in sorted((DATA / "fma").iterdir()):
            if p.is_file():
                # show against the 22 GiB FMA-medium expected size when it's the medium variant
                size = p.stat().st_size
                hint = ""
                if "medium" in p.name:
                    expected = 23_647_881_188
                    hint = f" ({100*size/expected:.0f}% of {fmt_bytes(expected)})"
                print(f"  fma/{p.name}: {fmt_bytes(size)}{hint}")

    print()
    print("=== disk free ===")
    total, used, free = shutil.disk_usage("/Users/prita")
    print(f"  free: {fmt_bytes(free)} of {fmt_bytes(total)} ({100*free/total:.0f}%)")


if __name__ == "__main__":
    main()
