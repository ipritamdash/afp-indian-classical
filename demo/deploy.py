"""Deploy a clone of the public demo Space (Tachyeon/afp-indian-classical-demo)
to your own Hugging Face account.

What this does:
  1. Verifies your HF token has the required permissions
  2. Snapshots the upstream public demo Space (~537 MB; uses HF's CDN, fast)
  3. Creates a new Space under your account
  4. Uploads the snapshot to your Space
  5. Waits for the build + prints the live URL

Usage:
  # one-shot
  HF_TOKEN=<your_token> python deploy.py --repo-name my-afp-demo
  # or explicit user
  HF_TOKEN=<your_token> python deploy.py --user <your_hf_username> --repo-name my-afp-demo

You'll need an HF token (https://huggingface.co/settings/tokens) with `write` scope.

Output:
  Space URL such as https://huggingface.co/spaces/<your_user>/my-afp-demo
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

UPSTREAM = "Tachyeon/afp-indian-classical-demo"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default=UPSTREAM,
                    help="The public Space to clone (default: %(default)s)")
    ap.add_argument("--repo-name", required=True,
                    help="Name of the new Space under your account (e.g., 'my-afp-demo')")
    ap.add_argument("--user", default=None,
                    help="Your HF username/org (defaults to whoami)")
    ap.add_argument("--private", action="store_true",
                    help="Create as a private Space (default: public)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        # Try the standard CLI-login cache file
        cache = Path.home() / ".cache" / "huggingface" / "token"
        if cache.exists():
            token = cache.read_text().strip()
    if not token:
        print("ERROR: no HF token found.")
        print("  Set HF_TOKEN env var, or run `huggingface-cli login` first.")
        print("  Get a token at https://huggingface.co/settings/tokens (need write scope).")
        return 1

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Install:  pip install 'huggingface_hub>=0.30'")
        return 1

    api = HfApi(token=token)

    # 1. Verify token + identity
    print(f"[1/5] verifying token...")
    me = api.whoami()
    print(f"  authed as: {me['name']}  (email: {me.get('email')})")
    owner = args.user or me["name"]
    target = f"{owner}/{args.repo_name}"
    print(f"  target Space: {target}  (private={args.private})")

    if args.dry_run:
        print("\n--dry-run: skipping snapshot + create + upload")
        return 0

    # 2. Snapshot upstream public Space
    print(f"\n[2/5] snapshotting upstream {args.upstream} ...")
    t0 = time.time()
    snap_dir = tempfile.mkdtemp(prefix="afp_demo_snap_")
    snapshot_download(
        repo_id=args.upstream,
        repo_type="space",
        local_dir=snap_dir,
        token=token,
    )
    n_files = sum(1 for _ in Path(snap_dir).rglob("*") if _.is_file())
    total_mb = sum(p.stat().st_size for p in Path(snap_dir).rglob("*") if p.is_file()) / 1e6
    print(f"  pulled {n_files} files, {total_mb:.1f} MB in {time.time()-t0:.0f}s")

    # 3. Create target Space
    print(f"\n[3/5] creating Space {target} ...")
    repo_url = api.create_repo(
        repo_id=target,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    print(f"  created: {repo_url}")

    # 4. Upload everything
    print(f"\n[4/5] uploading snapshot → {target} (LFS auto-handled) ...")
    t0 = time.time()
    commit = api.upload_folder(
        folder_path=snap_dir,
        repo_id=target,
        repo_type="space",
        commit_message=f"Deploy clone of {args.upstream}",
        ignore_patterns=[".git", ".git/**", "__pycache__", "*.pyc"],
    )
    print(f"  pushed in {time.time()-t0:.0f}s")
    print(f"  commit: {commit}")

    # 5. Wait for build
    print(f"\n[5/5] waiting for Space to build + start (typically 5-15 min)...")
    print(f"  monitor at: https://huggingface.co/spaces/{target}")
    print(f"  app file: app.py · sdk: gradio 6.14.0 · python 3.11")
    prev_stage = None
    start = time.time()
    while True:
        try:
            rt = api.get_space_runtime(repo_id=target)
            stage = rt.stage
        except Exception as e:
            print(f"  [warn] runtime fetch failed: {e}")
            stage = "UNKNOWN"
        if stage != prev_stage:
            print(f"  [{int(time.time()-start):4d}s] stage = {stage}")
            prev_stage = stage
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR"):
            break
        if time.time() - start > 1800:  # 30 min hard cap
            print("  [timeout] giving up after 30 minutes — check the Space UI directly")
            break
        time.sleep(20)

    if stage == "RUNNING":
        print(f"\n✓ DEPLOYED — live at https://huggingface.co/spaces/{target}")
        return 0
    else:
        print(f"\n✗ Build stuck at stage={stage}. Check logs:")
        print(f"  https://huggingface.co/spaces/{target}/logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
