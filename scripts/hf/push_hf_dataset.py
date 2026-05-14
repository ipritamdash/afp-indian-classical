"""Push hf_dataset/ to the Hub as a PRIVATE dataset.

Why private first: lets us inspect the rendered dataset card, run `load_dataset` against
it, and verify the AudioFolder pattern works end-to-end before any public exposure. Flip
to gated-public only at paper-submission time, via `api.update_repo_visibility(private=False)`
or the Hub web UI (Settings → Visibility).

Senior choices baked in:
- `huggingface_hub.HfApi` (Python) over the CLI: programmatic, returns commit refs.
- `upload_folder` with `commit_message` + `delete_patterns=["*"]` would wipe-and-replace
  on every push; we DON'T pass that — first push is a fresh repo, future pushes can
  use the `CommitOperationDelete` API to replace specific files.
- `allow_patterns` is left default (all files) — `.git*` etc. are auto-excluded by HfApi.
- Audio files are uploaded as-is (16k mono WAVs averaging ~315 KB) — no LFS gymnastics.
- We do NOT push token in plaintext anywhere; HfApi reads it from `~/.cache/huggingface/token`
  written by `huggingface-cli login` (or our staging step).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


PROJECT = Path(__file__).resolve().parent.parent.parent
STAGING = PROJECT / "hf_dataset"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Tachyeon",
                    help="HF username/org to push under (verified via whoami)")
    ap.add_argument("--repo-name", default="audio-fingerprint-indian-bench",
                    help="dataset repo name (will be created if missing)")
    ap.add_argument("--public", action="store_true",
                    help="push as PUBLIC instead of PRIVATE (default: private)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen, do not actually push")
    args = ap.parse_args()

    if not STAGING.exists():
        print(f"FATAL: staging dir missing — run scripts/hf/build_hf_dataset.py first")
        return 2

    api = HfApi()
    me = api.whoami()
    print(f"authed as: {me['name']}  (email={me.get('email')}, type={me.get('type')})")
    if me["name"].lower() != args.user.lower():
        print(f"WARN: whoami={me['name']!r} != --user={args.user!r}; using whoami value")
        owner = me["name"]
    else:
        owner = args.user

    repo_id = f"{owner}/{args.repo_name}"
    private = not args.public
    print(f"target repo: {repo_id}  (private={private})")

    # Pre-flight file count + size
    files = [p for p in STAGING.rglob("*") if p.is_file()]
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"staging: {len(files)} files, {total_mb:.1f} MB")

    if args.dry_run:
        print("\n--dry-run: skipping create_repo + upload_folder")
        return 0

    # 1) Create (idempotent — exist_ok=True)
    print(f"\n[1/2] create_repo({repo_id!r}, type=dataset, private={private})")
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    print(f"  repo URL: {url}")

    # 2) Upload folder — one commit, all artefacts
    print(f"\n[2/2] upload_folder({STAGING} → {repo_id})")
    commit_info = api.upload_folder(
        folder_path=str(STAGING),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=(
            "v0.6: 5 systems × 4 lengths × main+ablation + recipe v3 (3 seeds, "
            "Bonferroni-significant on 1s queries) + 2 pre-registered negative results"
        ),
        commit_description=(
            "MAJOR RELEASE.\n\n"
            "New in v0.6:\n"
            "- Added NMFP-ckpt-100 (Araz et al. ISMIR 2025) as system #5 — establishes "
            "  HR@1=1.000 ceiling across all 8 cells\n"
            "- Added Recipe v3 training-recipe improvement (3 seeds × 30 ep × BSZ=320):\n"
            "  pre-registered Bonferroni-significant gain on main_1s (pooled p=3.18e-06) "
            "  and ablation_1s (pooled p=0.0046), no cell regresses\n"
            "- Added pre-registered negative results: Intervention 2 (per-artist mean "
            "  subtraction) and hubness post-processing (InvSoftmax + CSLS)\n"
            "- All 4 query lengths (1/3/5/10 s) now exposed per (system × split)\n"
            "- 64 scores.json + 40+ result parquets + recipe_v3_pooled_mcnemar.csv + "
            "  3 protocol/results markdowns for full transparency\n"
            "- README rewritten\n\n"
            "Saraga 1.5 audio NOT redistributed (CC-BY-NC-SA 4.0; fetch via Zenodo). "
            "NMFP weights NOT redistributed (GPLv3/AGPLv3; fetch via Zenodo 15719945)."
        ),
        ignore_patterns=[".DS_Store", "*.tmp", "__pycache__"],
    )
    print(f"  commit: {commit_info}")

    # 3) Sanity: list a few files from the Hub to confirm
    print(f"\n=== first 20 files now visible at {repo_id} ===")
    listed = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    for f in listed[:20]:
        print(f"  {f}")
    print(f"  …({len(listed)} total)")

    print(f"\n✓ pushed.  Visit: https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
