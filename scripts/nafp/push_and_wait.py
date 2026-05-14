"""Push NAFP training kernel to Kaggle, poll until it completes, download outputs.

Reads scripts/nafp/{kernel-metadata.json, kaggle_train.py} and uses the kaggle
CLI under our access_token-authenticated user account.

Usage:
    uv run python scripts/nafp/push_and_wait.py
    uv run python scripts/nafp/push_and_wait.py --skip-push    # just poll + download
    uv run python scripts/nafp/push_and_wait.py --download-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NAFP_DIR = PROJECT_ROOT / "scripts" / "nafp"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "nafp" / "kaggle_output"


def run(cmd, *, check=True, capture=False) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def get_kernel_id() -> str:
    meta = json.loads((NAFP_DIR / "kernel-metadata.json").read_text())
    return meta["id"]


def push_kernel() -> str:
    """Upload the kernel + trigger run. Returns the kernel id."""
    print("=== pushing kernel to Kaggle ===", flush=True)
    run(["kaggle", "kernels", "push", "-p", str(NAFP_DIR)])
    return get_kernel_id()


def kernel_status(kernel_id: str) -> str:
    """Poll status. Returns one of: queued, running, complete, error, cancelRequested, cancelAcknowledged, cancelled."""
    r = run(["kaggle", "kernels", "status", kernel_id], capture=True)
    # Output is something like: 'aboutpritam/nafp-saraga-train has status "running"'
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if 'status "' in line:
            start = line.index('status "') + len('status "')
            end = line.index('"', start)
            return line[start:end]
    return "unknown:" + out.strip()[:200]


_TERMINAL_STATES = {"complete", "error", "cancelled", "cancelacknowledged"}


def _is_terminal(status: str) -> bool:
    # Kaggle returns "KernelWorkerStatus.RUNNING" / ".COMPLETE" / ".ERROR" etc.
    # Strip prefix and lowercase to compare.
    s = status.split(".")[-1].strip().lower()
    return s in _TERMINAL_STATES


def wait_for_completion(kernel_id: str, poll_sec: int = 60, max_wait_sec: int = 7200) -> str:
    print(f"=== waiting on kernel {kernel_id} (poll every {poll_sec}s, max {max_wait_sec}s) ===", flush=True)
    t0 = time.time()
    last_status = None
    while True:
        st = kernel_status(kernel_id)
        elapsed = int(time.time() - t0)
        if st != last_status:
            print(f"[{elapsed}s] status: {st}", flush=True)
            last_status = st
        if _is_terminal(st):
            return st
        if elapsed > max_wait_sec:
            print(f"[{elapsed}s] timeout after {max_wait_sec}s, last status={st}", flush=True)
            return "timeout"
        time.sleep(poll_sec)


def download_output(kernel_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== downloading kernel output to {out_dir} ===", flush=True)
    # Force overwrite via -o to avoid stale-data confusion
    run(["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir), "--force"])
    print("=== downloaded files ===", flush=True)
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}  ({p.stat().st_size:,} bytes)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-push", action="store_true", help="only poll + download (kernel already pushed)")
    ap.add_argument("--download-only", action="store_true", help="only download (kernel already complete)")
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--max-wait-sec", type=int, default=7200)
    args = ap.parse_args()

    kernel_id = get_kernel_id()

    if not args.skip_push and not args.download_only:
        push_kernel()

    if not args.download_only:
        final = wait_for_completion(kernel_id, args.poll_sec, args.max_wait_sec)
        print(f"=== final status: {final} ===", flush=True)
        if final not in ("complete",):
            print("Kernel didn't finish cleanly; downloading whatever output exists", flush=True)

    download_output(kernel_id, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
