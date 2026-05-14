"""Build reference + query manifests and cut query WAVs for a corpus.

Reproducibility contract:
  * Every cut offset is derived from a single GLOBAL_SEED + deterministic per-track seed
  * Manifest records exact offset_sec, length_sec, seed, source MD5
  * Cut WAVs are validated: duration == LONGEST_QUERY_SEC ± 1 sample, no NaN
  * Resamples to 16 kHz mono PCM_16 (canonical AFP query format)
  * Skips first SKIP_HEAD_SEC and last SKIP_TAIL_SEC of every source (silence/applause)
  * Requires source track ≥ MIN_TRACK_SEC; shorter tracks are excluded (logged)
  * Held-out test-source tracks are stratified by raga family across the corpus

Usage:
    uv run python scripts/build_testset.py hindustani
    uv run python scripts/build_testset.py carnatic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mirdata
import numpy as np
import pandas as pd
import soundfile as sf

# Reproducibility seed lives in manifest.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import (  # noqa: E402
    GLOBAL_SEED,
    REF_COLUMNS,
    QUERY_COLUMNS,
    RefEntry,
    QueryEntry,
)


# ── Configuration (locked, do not change between runs without re-versioning) ─
SKIP_HEAD_SEC = 10.0
SKIP_TAIL_SEC = 10.0
MIN_TRACK_SEC = 90.0           # ensures 5 × 10 s queries can be non-overlapping with margin
QUERIES_PER_TRACK = 5
LONGEST_QUERY_SEC = 10.0
JITTER_SEC = 0.5
TARGET_SR = 16_000
TARGET_CHANNELS = 1
TEST_SOURCE_N = 100            # how many tracks become query-source tracks
MANIFEST_VERSION = "1.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME_ROOT = PROJECT_ROOT / "data" / "mirdata"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
QUERIES_DIR = PROJECT_ROOT / "data" / "queries"


def md5_path(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def per_track_seed(global_seed: int, track_idx: int) -> int:
    """Deterministic per-track sub-seed. Independent of track_id text encoding."""
    h = hashlib.sha256(f"{global_seed}:{track_idx}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def list_field(meta_value) -> str | None:
    """Flatten Saraga's list-of-dicts metadata into a `; ` joined string.
    Handles nested `artist.name` (artists field) as well as flat name/title/common_name."""
    if meta_value is None:
        return None
    if isinstance(meta_value, (list, tuple)):
        names = []
        for item in meta_value:
            if isinstance(item, dict):
                nested = item.get("artist")
                if isinstance(nested, dict) and (nested.get("name") or nested.get("title")):
                    names.append(str(nested.get("name") or nested.get("title")).strip())
                    continue
                n = item.get("name") or item.get("title") or item.get("common_name") or item.get("mbid")
                if n:
                    names.append(str(n).strip())
            elif item is not None:
                names.append(str(item).strip())
        return "; ".join(names) or None
    if isinstance(meta_value, dict):
        nested = meta_value.get("artist")
        if isinstance(nested, dict) and (nested.get("name") or nested.get("title")):
            return str(nested.get("name") or nested.get("title")).strip()
        n = meta_value.get("name") or meta_value.get("title") or meta_value.get("common_name") or meta_value.get("mbid")
        return str(n).strip() if n else None
    s = str(meta_value).strip()
    return s or None


def safe_get(track, name):
    try:
        v = getattr(track, name)
        return v
    except Exception:
        return None


def meta_get(track, *keys):
    """Read from track.metadata (dict) trying each key spelling in order.
    Saraga Hindustani uses {raags, taals, layas}; Carnatic uses {raaga, taala, …}."""
    m = safe_get(track, "metadata")
    if isinstance(m, dict):
        for k in keys:
            v = m.get(k)
            if v is not None:
                return v
    for k in keys:
        v = safe_get(track, k)
        if v is not None:
            return v
    return None


def work_mbids_from(meta_value) -> str | None:
    if isinstance(meta_value, (list, tuple)):
        mbids = [str(d.get("mbid")).strip() for d in meta_value if isinstance(d, dict) and d.get("mbid")]
        return "; ".join(mbids) or None
    if isinstance(meta_value, dict) and meta_value.get("mbid"):
        return str(meta_value["mbid"]).strip()
    return None


def load_segment(audio_path: str, offset_sec: float, length_sec: float,
                 target_sr: int, target_channels: int) -> np.ndarray:
    """Read a precise audio window and resample to target_sr mono.
    Uses soundfile partial read; falls back to a single librosa.load if needed."""
    with sf.SoundFile(audio_path) as f:
        orig_sr = f.samplerate
        start = int(round(offset_sec * orig_sr))
        frames = int(round(length_sec * orig_sr))
        f.seek(start)
        data = f.read(frames=frames, dtype="float32", always_2d=True)  # (frames, channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # Downmix to mono
    if target_channels == 1 and data.shape[1] > 1:
        mono = data.mean(axis=1)
    else:
        mono = data[:, 0]
    # Resample if needed
    if orig_sr != target_sr:
        import librosa  # noqa: WPS433 — keep import lazy to speed up help text
        mono = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
    return mono.astype(np.float32, copy=False)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def validate_wav(path: Path, expected_len_sec: float, target_sr: int) -> None:
    info = sf.info(str(path))
    expected_frames = int(round(expected_len_sec * target_sr))
    actual_frames = info.frames
    if abs(actual_frames - expected_frames) > 1:
        raise ValueError(
            f"{path}: expected {expected_frames} frames, got {actual_frames}"
        )
    if info.samplerate != target_sr:
        raise ValueError(f"{path}: samplerate {info.samplerate} != {target_sr}")
    if info.channels != TARGET_CHANNELS:
        raise ValueError(f"{path}: channels {info.channels} != {TARGET_CHANNELS}")
    # NaN/Inf check on a quick read
    y, _ = sf.read(str(path), dtype="float32")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{path}: contains NaN/Inf samples")


def stratified_sample_indices(rng: random.Random, group_keys: list[str], n: int) -> list[int]:
    """Stratified sample by group_keys (e.g., raga). If a group has >1 candidate,
    proportionally sample from it. Falls back to uniform random when stratification
    cannot fill n (e.g., very many singleton groups)."""
    by_group: dict[str, list[int]] = {}
    for i, k in enumerate(group_keys):
        by_group.setdefault(k or "_none_", []).append(i)
    chosen: list[int] = []
    groups = sorted(by_group.keys())
    rng_groups = list(groups)
    rng.shuffle(rng_groups)
    # Round-robin pick
    while len(chosen) < n and rng_groups:
        for g in list(rng_groups):
            if not by_group[g]:
                rng_groups.remove(g)
                continue
            pick = rng.choice(by_group[g])
            by_group[g].remove(pick)
            chosen.append(pick)
            if len(chosen) >= n:
                break
    return chosen


def build(corpus: str) -> int:
    if corpus not in ("hindustani", "carnatic"):
        print(f"ERROR: corpus must be hindustani or carnatic, got {corpus!r}")
        return 2

    out_dir = MANIFEST_DIR / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_root = QUERIES_DIR / corpus
    queries_root.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "build.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("build")

    dataset_name = f"saraga_{corpus}"
    data_home = DATA_HOME_ROOT / dataset_name
    if not data_home.exists():
        log.error(f"data_home {data_home} does not exist — run download first")
        return 3

    log.info(f"loading mirdata {dataset_name} from {data_home}")
    ds = mirdata.initialize(dataset_name, data_home=str(data_home))
    track_ids = ds.track_ids
    log.info(f"loaded {len(track_ids)} track ids")

    # ── 1) Build the reference table (header inspect, no audio load) ────────
    refs: list[RefEntry] = []
    audio_paths: dict[str, str] = {}
    durations: dict[str, float] = {}
    raagas_for_strat: dict[str, str] = {}
    skipped: list[tuple[str, str]] = []

    for tid in track_ids:
        try:
            t = ds.track(tid)
            ap = safe_get(t, "audio_path")
            if not ap or not Path(ap).exists():
                skipped.append((tid, "no audio file"))
                continue
            info = sf.info(ap)
            dur = float(info.duration)
            if not np.isfinite(dur) or dur <= 0:
                skipped.append((tid, f"bad duration {dur}"))
                continue
            audio_paths[tid] = ap
            durations[tid] = dur

            raagas = list_field(meta_get(t, "raags", "raaga", "raagas"))
            raagas_for_strat[tid] = (raagas or "").split(";")[0].strip().lower() or "_unknown_"
            works_meta = meta_get(t, "works", "work")
            ref = RefEntry(
                ref_id=f"{corpus}_{tid}",
                corpus=corpus,
                source_track_id=tid,
                audio_path=ap,
                duration_sec=round(dur, 3),
                samplerate=info.samplerate,
                channels=info.channels,
                raagas=raagas,
                taalas=list_field(meta_get(t, "taals", "taala", "taalas")),
                artists=list_field(meta_get(t, "album_artists", "artists")),
                works=list_field(works_meta),
                work_mbids=work_mbids_from(works_meta),
                role="library_only",
            )
            refs.append(ref)
        except Exception as e:
            skipped.append((tid, f"exception: {e}"))

    log.info(f"refs built: {len(refs)} OK, {len(skipped)} skipped")
    for tid, reason in skipped[:20]:
        log.warning(f"skipped {tid}: {reason}")

    # ── 2) Pick test-source tracks (stratified, only tracks with ≥MIN_TRACK_SEC) ──
    eligible_idx = [i for i, r in enumerate(refs) if r.duration_sec >= MIN_TRACK_SEC]
    log.info(f"eligible for queries (≥ {MIN_TRACK_SEC}s): {len(eligible_idx)}")
    if len(eligible_idx) == 0:
        log.error("No eligible tracks — corpus too short or empty")
        return 4

    rng = random.Random(GLOBAL_SEED)
    eligible_groups = [raagas_for_strat[refs[i].source_track_id] for i in eligible_idx]
    n_to_pick = min(TEST_SOURCE_N, len(eligible_idx))
    chosen_within = stratified_sample_indices(rng, eligible_groups, n_to_pick)
    test_source_track_ids = sorted({refs[eligible_idx[c]].source_track_id for c in chosen_within})
    log.info(f"test-source tracks selected: {len(test_source_track_ids)}")

    # Mark roles
    test_source_set = set(test_source_track_ids)
    for r in refs:
        if r.source_track_id in test_source_set:
            r.role = "library_and_test_source"

    # ── 3) Cut queries ──────────────────────────────────────────────────────
    queries: list[QueryEntry] = []
    src_md5: dict[str, str] = {}
    t_cut0 = time.time()
    for track_idx, tid in enumerate(sorted(test_source_track_ids)):
        ref = next(r for r in refs if r.source_track_id == tid)
        dur = ref.duration_sec
        seed = per_track_seed(GLOBAL_SEED, track_idx)
        rngt = random.Random(seed)

        usable_start = SKIP_HEAD_SEC
        usable_end = dur - SKIP_TAIL_SEC - LONGEST_QUERY_SEC
        span = usable_end - usable_start
        # linearly-spaced anchors in (usable_start, usable_end)
        anchors = [usable_start + (i + 1) / (QUERIES_PER_TRACK + 1) * span for i in range(QUERIES_PER_TRACK)]
        jitters = [rngt.uniform(-JITTER_SEC, JITTER_SEC) for _ in anchors]
        offsets = [max(usable_start, min(usable_end, round(a + j, 3))) for a, j in zip(anchors, jitters)]

        # Compute source MD5 once (small files matter; Saraga MP3s are 50-100 MB so do this lazily)
        if tid not in src_md5:
            src_md5[tid] = md5_path(Path(ref.audio_path))

        for qi, off in enumerate(offsets):
            qid = f"{corpus}_t{track_idx:04d}_q{qi}"
            wav_path = queries_root / f"{qid}.wav"
            try:
                audio = load_segment(ref.audio_path, off, LONGEST_QUERY_SEC, TARGET_SR, TARGET_CHANNELS)
                write_wav(wav_path, audio, TARGET_SR)
                validate_wav(wav_path, LONGEST_QUERY_SEC, TARGET_SR)
            except Exception as e:
                log.error(f"failed cut for {qid} at {off:.2f}s of {ref.audio_path}: {e}")
                continue
            queries.append(
                QueryEntry(
                    query_id=qid,
                    ref_id=ref.ref_id,
                    corpus=corpus,
                    audio_path=str(wav_path),
                    offset_sec=float(off),
                    length_sec=LONGEST_QUERY_SEC,
                    seed=seed,
                )
            )

        if (track_idx + 1) % 10 == 0:
            log.info(f"cut {(track_idx+1)*QUERIES_PER_TRACK}/{n_to_pick*QUERIES_PER_TRACK} queries…")

    log.info(f"queries cut: {len(queries)} in {time.time()-t_cut0:.1f}s")

    # ── 4) Write manifests ──────────────────────────────────────────────────
    refs_df = pd.DataFrame([asdict(r) for r in refs], columns=REF_COLUMNS)
    queries_df = pd.DataFrame([asdict(q) for q in queries], columns=QUERY_COLUMNS)

    # Attach source MD5 to refs
    refs_df["source_md5"] = refs_df["source_track_id"].map(src_md5).fillna("")

    refs_csv = out_dir / "refs.csv"
    queries_csv = out_dir / "queries.csv"
    refs_df.to_csv(refs_csv, index=False)
    queries_df.to_csv(queries_csv, index=False)

    # ── 5) Persist the locked config alongside the manifests ────────────────
    config_path = out_dir / "config.json"
    config = {
        "manifest_version": MANIFEST_VERSION,
        "global_seed": GLOBAL_SEED,
        "skip_head_sec": SKIP_HEAD_SEC,
        "skip_tail_sec": SKIP_TAIL_SEC,
        "min_track_sec": MIN_TRACK_SEC,
        "queries_per_track": QUERIES_PER_TRACK,
        "longest_query_sec": LONGEST_QUERY_SEC,
        "jitter_sec": JITTER_SEC,
        "target_sr": TARGET_SR,
        "target_channels": TARGET_CHANNELS,
        "test_source_n": TEST_SOURCE_N,
        "n_refs": len(refs),
        "n_test_source": len(test_source_track_ids),
        "n_queries": len(queries),
        "n_skipped": len(skipped),
        "corpus": corpus,
        "dataset_name": dataset_name,
    }
    config_path.write_text(json.dumps(config, indent=2))

    log.info(f"WROTE {refs_csv}  ({len(refs_df)} rows)")
    log.info(f"WROTE {queries_csv}  ({len(queries_df)} rows)")
    log.info(f"WROTE {config_path}")
    log.info(f"queries WAV dir: {queries_root}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["hindustani", "carnatic"])
    args = ap.parse_args()
    try:
        return build(args.corpus)
    except Exception:
        import traceback
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
