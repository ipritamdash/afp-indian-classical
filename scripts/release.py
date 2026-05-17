"""End-to-end release orchestrator for the AFP-Indian-Classical project.

One command from the repo root rebuilds every public-facing artefact and pushes
the lot. Idempotent. Each subcommand is independently runnable.

USAGE
-----
    /tmp/venv_metal/bin/python scripts/release.py all              # full pipeline
    /tmp/venv_metal/bin/python scripts/release.py pdf              # build teaching PDFs only
    /tmp/venv_metal/bin/python scripts/release.py hf-card          # push HF dataset README only
    /tmp/venv_metal/bin/python scripts/release.py hf-verify        # poll preview server
    /tmp/venv_metal/bin/python scripts/release.py github "msg"     # commit + push to GitHub

DESIGN NOTES
------------
- `hf-card` pushes only README.md by default — safe to re-run, never touches data
- `hf-full` is a separate command for re-uploading every staging file (rare; slow)
- `hf-verify` is idempotent and polls; safe to run anytime
- `github` uses Aryan Banwala as commit author (matches the project's other commits)
- All steps short-circuit cleanly if there's nothing to do
- No secrets in this file; reads HF_TOKEN from env or ~/.cache/huggingface/token
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HF_REPO = "Tachyeon/audio-fingerprint-indian-bench"
HF_README_LOCAL = REPO / "hf_dataset" / "README.md"

EXPECTED_ROWS = {
    ("refs", "train"): 357,
    ("inspection_tracks", "train"): 357,
    ("inspection_sections", "train"): 749,
    ("inspection_works", "train"): 616,
    ("inspection_leakage_pairs", "train"): 123,
    ("queries", "test"): 1000,
    ("queries_ablation", "test"): 632,
}

GIT_AUTHOR_NAME = "Aryan Banwala"
GIT_AUTHOR_EMAIL = "hemrajyadav917@gmail.com"

# ---------- pretty-printing ----------

def hr(title: str) -> None:
    print(f"\n{'─' * 72}\n{title}\n{'─' * 72}")


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}", file=sys.stderr)


# ---------- subcommands ----------

def cmd_pdf(args: argparse.Namespace) -> int:
    """Rebuild both teaching PDFs from their markdown sources."""
    hr("[1] Rebuild teaching PDFs")
    py = sys.executable
    script = REPO / "scripts" / "build_teach_me_pdf.py"
    if not script.exists():
        fail(f"missing: {script}")
        return 2
    for doc in ("teach", "overview"):
        r = subprocess.run([py, str(script), "--doc", doc])
        if r.returncode != 0:
            fail(f"PDF build failed for --doc {doc}")
            return r.returncode
        ok(f"built --doc {doc}")
    return 0


def cmd_hf_card(args: argparse.Namespace) -> int:
    """Push only README.md to the HF dataset. Safe, fast, no data re-upload."""
    hr("[2] Push HF dataset card (README.md only)")
    if not HF_README_LOCAL.exists():
        fail(f"missing local card: {HF_README_LOCAL}")
        return 2

    local_sha = hashlib.sha256(HF_README_LOCAL.read_bytes()).hexdigest()
    print(f"  local README sha256[:12]: {local_sha[:12]}")

    if args.dry_run:
        warn("--dry-run: not pushing")
        return 0

    try:
        from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    except ImportError:
        fail("huggingface_hub not installed in this Python")
        return 2

    api = HfApi()
    # Skip if remote already matches local
    try:
        remote_path = hf_hub_download(HF_REPO, "README.md", repo_type="dataset")
        remote_sha = hashlib.sha256(Path(remote_path).read_bytes()).hexdigest()
        print(f"  live README sha256[:12]: {remote_sha[:12]}")
        if remote_sha == local_sha:
            ok("live README already matches local — nothing to push")
            return 0
    except Exception as e:
        warn(f"could not fetch live README for comparison: {e}")

    res = api.create_commit(
        repo_id=HF_REPO,
        repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo="README.md",
                                        path_or_fileobj=str(HF_README_LOCAL))],
        commit_message=args.message or "Update dataset card (automated release)",
    )
    ok(f"pushed commit: {res.oid[:12]}")
    ok(f"url: {res.commit_url}")
    return 0


def cmd_hf_verify(args: argparse.Namespace) -> int:
    """Poll the HF preview server until every config reports its expected row count."""
    hr("[3] Verify HF preview-server row counts")
    deadline = time.time() + args.timeout
    last_seen: dict[tuple[str, str], int | None] = {k: None for k in EXPECTED_ROWS}
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            url = f"https://datasets-server.huggingface.co/size?dataset={HF_REPO}"
            with urllib.request.urlopen(url, timeout=20) as r:
                res = json.loads(r.read())
            for s in res.get("size", {}).get("splits", []):
                last_seen[(s["config"], s["split"])] = s.get("num_rows")
        except Exception as e:
            warn(f"attempt {attempts}: {e}")
            time.sleep(args.interval)
            continue

        wrong = [
            (cfg, sp, last_seen[(cfg, sp)], exp)
            for (cfg, sp), exp in EXPECTED_ROWS.items()
            if last_seen[(cfg, sp)] != exp
        ]
        if not wrong:
            ok(f"all 7 configs report expected rows (after {attempts} polls)")
            for (cfg, sp), exp in EXPECTED_ROWS.items():
                print(f"    cfg={cfg:30s} split={sp:6s} rows={exp}")
            return 0

        # Still wrong — print drift, then retry
        msg = ", ".join(f"{cfg}={last_seen[(cfg,sp)]!s}(exp {exp})" for cfg, sp, _, exp in wrong)
        print(f"  poll {attempts}: {msg}")
        time.sleep(args.interval)

    fail(f"preview server did not converge within {args.timeout}s")
    print("  last seen:")
    for (cfg, sp), exp in EXPECTED_ROWS.items():
        cur = last_seen[(cfg, sp)]
        marker = "✓" if cur == exp else "✗"
        print(f"    {marker} cfg={cfg:30s} split={sp:6s} rows={cur!s} (expected {exp})")
    return 1


def cmd_github(args: argparse.Namespace) -> int:
    """Commit and push any pending changes to GitHub."""
    hr("[4] Commit + push to GitHub")

    # 1) Check git state
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                            capture_output=True, text=True)
    if status.returncode != 0:
        fail(f"git status failed: {status.stderr.strip()}")
        return status.returncode

    pending = status.stdout.strip()
    if not pending:
        ok("git working tree clean — nothing to commit")
        # Still try to push in case local is ahead of origin
        push = subprocess.run(["git", "-C", str(REPO), "push", "origin", "main"],
                              capture_output=True, text=True)
        if "Everything up-to-date" in push.stderr or "Everything up-to-date" in push.stdout:
            ok("origin/main already up to date")
        elif push.returncode == 0:
            ok("pushed pre-existing local commits to origin/main")
        else:
            warn(f"git push reported: {push.stderr.strip()}")
        return 0

    if args.dry_run:
        warn("--dry-run: not committing or pushing")
        print("  pending changes:")
        for line in pending.splitlines():
            print(f"    {line}")
        return 0

    # 2) Stage the files release.py expects to touch
    files_to_add = [
        "docs/AFP_Overview_Guide.pdf",
        "docs/AFP_Teaching_Guide.pdf",
        "docs/overview.md",
        "docs/teach_me.md",
        "hf_dataset/README.md",
        "README.md",
        "scripts/release.py",
        "scripts/build_teach_me_pdf.py",
    ]
    existing = [f for f in files_to_add if (REPO / f).exists()]
    subprocess.run(["git", "-C", str(REPO), "add", *existing], check=False)

    # If after staging there's nothing, bail
    staged = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--name-only"],
                            capture_output=True, text=True).stdout.strip()
    if not staged:
        ok("nothing relevant staged after add (other changes were not part of release)")
        return 0

    # 3) Commit + push (author = Aryan Banwala, no --no-verify)
    msg = args.message or "Automated release sync"
    body = (
        "Generated by scripts/release.py.\n\n"
        f"Includes any of: rebuilt teaching PDFs, updated HF dataset card, "
        f"release-script changes.\n\n"
        "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
    )
    full_msg = f"{msg}\n\n{body}"
    commit = subprocess.run(
        ["git", "-C", str(REPO),
         "-c", f"user.name={GIT_AUTHOR_NAME}",
         "-c", f"user.email={GIT_AUTHOR_EMAIL}",
         "commit", "-m", full_msg],
        capture_output=True, text=True,
    )
    if commit.returncode != 0:
        if "nothing to commit" in commit.stdout + commit.stderr:
            ok("nothing to commit after staging")
            return 0
        fail(f"git commit failed:\n{commit.stdout}\n{commit.stderr}")
        return commit.returncode
    ok(f"committed:\n    {commit.stdout.splitlines()[0]}")

    push = subprocess.run(["git", "-C", str(REPO), "push", "origin", "main"],
                          capture_output=True, text=True)
    if push.returncode != 0:
        fail(f"git push failed:\n{push.stdout}\n{push.stderr}")
        return push.returncode
    ok("pushed to origin/main")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Full pipeline: PDFs → HF card → preview verify → GitHub sync."""
    print(f"\n=== Release pipeline @ {datetime.datetime.now().isoformat(timespec='seconds')} ===")
    for fn in (cmd_pdf, cmd_hf_card, cmd_hf_verify, cmd_github):
        rc = fn(args)
        if rc != 0:
            fail(f"step `{fn.__name__}` returned {rc}; aborting pipeline")
            return rc
    hr("DONE")
    ok(f"HF dataset: https://huggingface.co/datasets/{HF_REPO}")
    ok(f"HF demo:    https://huggingface.co/spaces/Tachyeon/afp-indian-classical-demo")
    ok(f"GitHub:     https://github.com/ipritamdash/afp-indian-classical")
    return 0


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pdf", help="rebuild both teaching PDFs")
    sp.set_defaults(func=cmd_pdf)

    sp = sub.add_parser("hf-card", help="push README.md to the HF dataset (idempotent)")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("-m", "--message", default=None, help="commit message")
    sp.set_defaults(func=cmd_hf_card)

    sp = sub.add_parser("hf-verify", help="poll preview-server until row counts are correct")
    sp.add_argument("--interval", type=int, default=8, help="seconds between polls")
    sp.add_argument("--timeout", type=int, default=300, help="give up after N seconds")
    sp.set_defaults(func=cmd_hf_verify)

    sp = sub.add_parser("github", help="commit + push release artefacts to GitHub")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("-m", "--message", default=None,
                    help="commit message (defaults to 'Automated release sync')")
    sp.set_defaults(func=cmd_github)

    sp = sub.add_parser("all", help="run full pipeline: pdf → hf-card → hf-verify → github")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("-m", "--message", default=None)
    sp.add_argument("--interval", type=int, default=8)
    sp.add_argument("--timeout", type=int, default=300)
    sp.set_defaults(func=cmd_all)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
