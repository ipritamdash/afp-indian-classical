"""NAFP training on Kaggle GPU — v5 REAL TRAINING with fail-fast diagnostics.

Real 10-epoch run, NOT a toy. Diagnostics are wired in such that the kernel exits in <90 s
if GPU is unhealthy (instead of silently running on CPU for ~20 h):

  - REDUCE_ITEMS_P=0     → all 10 000 training files
  - MAX_EPOCH=10         → ICASSP-paper-style training length
  - PYTHONUNBUFFERED + sys.stdout.reconfigure(line_buffering=True) so per-step prints reach
    Kaggle's log immediately (Keras `Progbar` uses `\r` which Kaggle's log capture buffers)
  - HARD GPU assertion: list_physical_devices('GPU') must be non-empty (sys.exit(7))
  - HARD GPU COMPUTE assertion: a 2048×2048 matmul on /GPU:0 must finish in < 200 ms
    (healthy T4/P100: < 30 ms; CPU fallback: > 500 ms) — sys.exit(8) otherwise
  - trainer.py monkey-patched to print every 5 train steps + every 10 val steps with
    flush=True, including running avg step-time so CPU fallback is obvious within 60 s

Expected timing on a healthy T4:
  - GPU matmul check: < 30 ms
  - per training step: 1–3 s with TR_BATCH_SZ=120
  - ~167 steps/epoch × 2 s ≈ 5–6 min/epoch → 10 epochs ≈ 55–90 min wall-clock

The 1 h timeout of the v4 diagnostic is raised to 11.5 h (Kaggle's hard kernel limit is 12 h).

Authoritative source check (verified 2026-05-12 against /tmp/nafp_wheels/neural-audio-fp/):
- run.py line 14: config loaded from `./config/<CONFIG_NAME>.yaml`
- config/default.yaml: DIR (6 fields), DATA_SEL.REDUCE_ITEMS_P, BSZ.TR_BATCH_SZ=120, TR_N_ANCHOR=60,
  TRAIN.MAX_EPOCH=100, TRAIN.MINI_TEST_IN_TRAIN=True, DEVICE.CPU_N_WORKERS=4
- model/trainer.py lines 187-193: the per-step loop (Progbar.add) → monkey-patch target
- kapre 0.3.5 pinned in requirements.txt; using 0.3.7 (only Py3.12 wheel — closest patch release,
  API-compatible for MelSpectrogram / STFT classes NAFP uses)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ── 0) Environment (must be set BEFORE any TF import) ──────────────────────
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Force line-buffered stdout/stderr so per-step prints show in Kaggle's log immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


# ── Paths (Kaggle-specific) ────────────────────────────────────────────────
WORKING = Path("/kaggle/working")
NAFP_REPO = WORKING / "neural-audio-fp"
CKPT_DIR = WORKING / "logs" / "checkpoint" / "pipeline"


def find_input_dataset() -> Path:
    """Locate the Mimbres dataset under /kaggle/input regardless of mount layout
    (API-pushed vs UI-uploaded kernels mount the dataset at different paths)."""
    root = Path("/kaggle/input")
    if not root.exists():
        raise SystemExit(f"FATAL: {root} does not exist — not running on Kaggle?")
    print(f"=== /kaggle/input layout ===", flush=True)
    for p in root.iterdir():
        print(f"  {p}", flush=True)
    candidates: list[Path] = []
    for path in root.rglob("aug"):
        if path.is_dir() and (path.parent / "music").is_dir():
            candidates.append(path.parent)
            if len(candidates) > 5:
                break
    if not candidates:
        raise SystemExit(
            "FATAL: could not find a dir with both `aug/` and `music/` under /kaggle/input/"
        )
    chosen = candidates[0]
    print(f"=== chosen INPUT_DATASET: {chosen} ===", flush=True)
    train_dir = chosen / "music" / "train-10k-30s"
    if not train_dir.exists():
        raise SystemExit(f"FATAL: training subdir missing: {train_dir}")
    n_train = len(list(train_dir.iterdir()))
    print(f"  music/train-10k-30s/ has {n_train} entries", flush=True)
    return chosen


def run(cmd, *, check=True, cwd=None, env=None, timeout=None):
    print(f"\n$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check,
                          cwd=cwd, env=env, timeout=timeout)


def patch_trainer_cosine_decay(repo_root: Path) -> None:
    """TF 2.18+ removed tf.keras.experimental.CosineDecay alias."""
    trainer = repo_root / "model" / "trainer.py"
    src = trainer.read_text()
    new = src.replace(
        "tf.keras.experimental.CosineDecay",
        "tf.keras.optimizers.schedules.CosineDecay",
    )
    if new != src:
        trainer.write_text(new)
        print(f"patched {trainer.name}: CosineDecay alias", flush=True)


def patch_trainer_step_logging(repo_root: Path) -> None:
    """Replace the silent per-step loop in trainer.py with one that prints every 5
    steps + flushes. Patches both the training loop (after line 187) and validation
    loop (after line 207). Verified target strings exist at the exact indentation
    shown — patch is a no-op if the strings change upstream (fails loudly via
    `if old not in src: raise`)."""
    trainer = repo_root / "model" / "trainer.py"
    src = trainer.read_text()

    # Training loop block — exact 8-space indent
    tr_old = (
        '        i = 0\n'
        '        while i < len(enq.sequence):\n'
        '            X = next(enq.get()) # X: Tuple(Xa, Xp)\n'
        '            avg_loss, sim_mtx = train_step(X, m_pre, m_specaug, m_fp,\n'
        '                                            loss_obj_train, helper)\n'
        '            progbar.add(1, values=[("tr loss", avg_loss)])\n'
        '            i += 1\n'
        '        enq.stop()'
    )
    tr_new = (
        '        # ── diagnostic patch (v5): per-step flushed prints ──\n'
        '        import time as _dt\n'
        '        _tr_t0 = _dt.time()\n'
        '        _tr_n = len(enq.sequence)\n'
        '        print(f"  [tr] starting epoch loop with {_tr_n} steps", flush=True)\n'
        '        i = 0\n'
        '        while i < _tr_n:\n'
        '            X = next(enq.get()) # X: Tuple(Xa, Xp)\n'
        '            avg_loss, sim_mtx = train_step(X, m_pre, m_specaug, m_fp,\n'
        '                                            loss_obj_train, helper)\n'
        '            progbar.add(1, values=[("tr loss", avg_loss)])\n'
        '            i += 1\n'
        '            if i <= 3 or i % 5 == 0 or i == _tr_n:\n'
        '                _el = _dt.time() - _tr_t0\n'
        '                print(f"    [tr step {i:>3}/{_tr_n}] loss={float(avg_loss):.4f}  '
        'avg={_el/i:.2f}s/step  elapsed={_el:.1f}s", flush=True)\n'
        '        enq.stop()'
    )
    if tr_old not in src:
        raise SystemExit("FATAL: training-loop patch target string not found in trainer.py")
    src = src.replace(tr_old, tr_new, 1)

    # Validation loop block — same shape
    val_old = (
        '        i = 0\n'
        '        while i < len(enq.sequence):\n'
        '            X = next(enq.get()) # X: Tuple(Xa, Xp)\n'
        '            _, sim_mtx = val_step(X, m_pre, m_fp, loss_obj_val,\n'
        '                                  helper)\n'
        '            i += 1\n'
        '        enq.stop()'
    )
    val_new = (
        '        # ── diagnostic patch (v5) ──\n'
        '        import time as _dt2\n'
        '        _val_t0 = _dt2.time()\n'
        '        _val_n = len(enq.sequence)\n'
        '        print(f"  [val] starting val loop with {_val_n} steps", flush=True)\n'
        '        i = 0\n'
        '        while i < _val_n:\n'
        '            X = next(enq.get()) # X: Tuple(Xa, Xp)\n'
        '            _, sim_mtx = val_step(X, m_pre, m_fp, loss_obj_val, helper)\n'
        '            i += 1\n'
        '            if i <= 2 or i % 10 == 0 or i == _val_n:\n'
        '                _el = _dt2.time() - _val_t0\n'
        '                print(f"    [val step {i:>3}/{_val_n}]  avg={_el/i:.2f}s/step  '
        'elapsed={_el:.1f}s", flush=True)\n'
        '        enq.stop()'
    )
    if val_old not in src:
        raise SystemExit("FATAL: validation-loop patch target string not found in trainer.py")
    src = src.replace(val_old, val_new, 1)

    trainer.write_text(src)
    print(f"patched {trainer.name}: per-step flushed logging (tr + val)", flush=True)


def write_pipeline_config(repo_root: Path, input_dataset: Path) -> Path:
    """Load default.yaml as a dict, override DIR + diagnostic knobs, write pipeline.yaml."""
    import yaml
    src_yaml = repo_root / "config" / "default.yaml"
    dst_yaml = repo_root / "config" / "pipeline.yaml"

    with src_yaml.open() as f:
        cfg = yaml.safe_load(f)

    base = str(input_dataset)
    cfg["DIR"]["SOURCE_ROOT_DIR"]  = f"{base}/music/"
    cfg["DIR"]["BG_ROOT_DIR"]      = f"{base}/aug/bg/"
    cfg["DIR"]["IR_ROOT_DIR"]      = f"{base}/aug/ir/"
    cfg["DIR"]["SPEECH_ROOT_DIR"]  = f"{base}/aug/speech/common_voice_8k/en/"
    cfg["DIR"]["OUTPUT_ROOT_DIR"]  = f"{WORKING}/logs/emb/"
    cfg["DIR"]["LOG_ROOT_DIR"]     = f"{WORKING}/logs/"

    # v5 REAL-TRAINING OVERRIDES
    cfg["DATA_SEL"]["REDUCE_ITEMS_P"] = 0   # 0 → use ALL training files (default.yaml's "inactivate")
    cfg["TRAIN"]["MINI_TEST_IN_TRAIN"] = False  # skip mid-train mini-search, we eval on Mac vs Saraga
    # MAX_EPOCH from default.yaml is 100; we pass --max_epoch=10 via CLI to override

    with dst_yaml.open("w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"wrote {dst_yaml}", flush=True)
    return dst_yaml


def main() -> int:
    t_start = time.time()
    print(f"=== NAFP v5 REAL TRAINING run @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

    # ── 1) Discover dataset ───────────────────────────────────────────────
    input_dataset = find_input_dataset()

    # ── 2) Offline-install bundled wheels ─────────────────────────────────
    wheels_dir = None
    for cand in (Path("/kaggle/input/nafp-wheels"),
                 Path("/kaggle/input/datasets/aboutpritam/nafp-wheels")):
        if cand.exists():
            wheels_dir = cand
            break
    if wheels_dir is None:
        print("FATAL: nafp-wheels dataset not attached", flush=True)
        return 2
    print(f"[deps] wheels at {wheels_dir}", flush=True)
    run([sys.executable, "-m", "pip", "install", "-q",
         "--no-index", "--find-links", str(wheels_dir),
         "kapre", "tf_keras"], timeout=180)

    # ── 3) Copy NAFP source (no internet needed) ──────────────────────────
    src_repo = wheels_dir / "neural-audio-fp"
    if not src_repo.exists():
        print(f"FATAL: bundled NAFP source missing at {src_repo}", flush=True)
        return 3
    if NAFP_REPO.exists():
        shutil.rmtree(NAFP_REPO)
    shutil.copytree(src_repo, NAFP_REPO)
    print(f"copied NAFP source → {NAFP_REPO}", flush=True)

    # ── 4) Patches ────────────────────────────────────────────────────────
    patch_trainer_cosine_decay(NAFP_REPO)
    patch_trainer_step_logging(NAFP_REPO)

    # ── 5) Pipeline config ────────────────────────────────────────────────
    write_pipeline_config(NAFP_REPO, input_dataset)

    # ── 6) GPU HARD-ASSERT (fail fast, no silent CPU fallback) ────────────
    import tensorflow as tf
    print(f"TF: {tf.__version__}", flush=True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible: {len(gpus)}: {gpus}", flush=True)
    if not gpus:
        print("FATAL[7]: NO GPU visible — kernel-metadata says enable_gpu=true but TF doesn't see it. "
              "Likely cuInit failed at TF import. Aborting before wasting compute.", flush=True)
        return 7
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print(f"  (set_memory_growth failed for {g}: {e})", flush=True)

    # 6b) GPU compute assertion — actually run a matmul on /GPU:0 and time it
    print("\nGPU compute check (2048×2048 matmul on /GPU:0)…", flush=True)
    with tf.device("/GPU:0"):
        a = tf.random.normal((2048, 2048))
        b = tf.random.normal((2048, 2048))
        # warmup
        _ = tf.matmul(a, b).numpy()
        t0 = time.time()
        c = tf.matmul(a, b)
        _ = c.numpy()
        gpu_matmul_ms = (time.time() - t0) * 1000
    print(f"GPU matmul: {gpu_matmul_ms:.1f} ms  "
          f"(healthy GPU: <30 ms, CPU fallback: >500 ms)", flush=True)
    if gpu_matmul_ms > 200:
        print(f"FATAL[8]: matmul too slow for a real GPU — TF is on CPU despite GPU device listing. "
              f"Aborting before wasting compute.", flush=True)
        return 8

    # ── 7) Run training (10 epochs, full dataset) ─────────────────────────
    epochs = int(os.environ.get("NAFP_TRAIN_EPOCHS", "10"))
    print(f"\n=== TRAIN: {epochs} epochs (FULL DATA) ===\n", flush=True)
    train_t0 = time.time()
    try:
        run(
            [sys.executable, "-u", "run.py", "train", "pipeline", "-c", "pipeline",
             f"--max_epoch={epochs}"],
            cwd=str(NAFP_REPO),
            timeout=41400,  # 11.5 h hard cap — Kaggle's kernel limit is 12 h
        )
    except subprocess.CalledProcessError as e:
        print(f"training exited non-zero: exit={e.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"FATAL[9]: training exceeded 11.5 h cap — kernel will be terminated by Kaggle", flush=True)
        return 9
    train_sec = time.time() - train_t0
    print(f"\n=== TRAIN finished in {train_sec/60:.2f} min ===\n", flush=True)

    # ── 8) Verify checkpoints ─────────────────────────────────────────────
    ckpts = []
    if CKPT_DIR.exists():
        ckpts = sorted(CKPT_DIR.glob("ckpt-*.index"))
        print(f"checkpoints: {[c.name for c in ckpts]}", flush=True)
    else:
        print(f"WARN: checkpoint dir missing: {CKPT_DIR}", flush=True)

    # ── 9) Manifest ───────────────────────────────────────────────────────
    manifest = {
        "version": "v5-real-training",
        "epochs_requested": epochs,
        "reduce_items_p": 0,
        "gpus_visible": len(gpus),
        "gpu_matmul_ms": round(gpu_matmul_ms, 1),
        "train_seconds": round(train_sec, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "tf_version": tf.__version__,
        "checkpoint_files": [c.name for c in ckpts],
    }
    (WORKING / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0 if ckpts else 6


if __name__ == "__main__":
    sys.exit(main())
