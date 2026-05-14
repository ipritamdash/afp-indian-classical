"""Cut shorter-length query variants (1 s / 3 s / 5 s) from the existing 10-s queries.

Why this instead of re-running `build_testset.py` with `LONGEST_QUERY_SEC=N`:
  - The 10-s queries were cut at sample-accurate offsets via soundfile + resampled
    to 16 kHz mono PCM_16 with `librosa.resample(..., res_type='soxr_hq')`.
  - Truncating the first N×TARGET_SR samples of an existing 10-s WAV is
    mathematically equivalent to re-cutting an N-sec query from the SAME source
    at the SAME offset — the resample is fully deterministic and operates on the
    same source samples.
  - This isolates the *length effect* from any offset-distribution shift that
    would happen if we re-ran build_testset with a different LONGEST_QUERY_SEC
    (because usable_end = dur - SKIP_TAIL_SEC - LONGEST_QUERY_SEC depends on it).

Produces:
  data/queries_1s/{hindustani,carnatic}/{*.wav, ablation/*.wav}
  data/queries_3s/...
  data/queries_5s/...
  data/manifests/{corpus}/queries_{1s,3s,5s}.csv
  data/manifests/{corpus}/queries_ablation_{1s,3s,5s}.csv

Validation:
  - Each output WAV has exactly N × 16000 samples (1 sample tolerance).
  - Each output WAV has SR=16000, channels=1, subtype=PCM_16.
  - Length manifest's `length_sec` reflects the new length; all other columns
    (offset_sec, seed, ref_id, query_id, etc.) are preserved unchanged.

Usage:
    uv run python scripts/build_query_length_variants.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MANIFEST = DATA / "manifests"
TARGET_SR = 16_000
TARGET_CHANNELS = 1
TARGET_LENGTHS_SEC = (1.0, 3.0, 5.0)

CORPORA = ("hindustani", "carnatic")
VARIANTS = (
    # (manifest_input_csv, manifest_suffix, queries_subdir)
    ("queries.csv",          "queries",          ""),         # main: no subdir
    ("queries_ablation.csv", "queries_ablation", "ablation"), # ablation under queries/{corpus}/ablation/
)


def truncate_wav(src_path: Path, dst_path: Path, n_samples: int) -> int:
    """Read first n_samples of src (mono float32), write to dst as PCM_16 16kHz.
    Returns the number of samples actually written. Raises if src is shorter."""
    with sf.SoundFile(src_path) as f:
        if f.samplerate != TARGET_SR:
            raise ValueError(f"{src_path}: unexpected sr={f.samplerate}")
        if f.channels != TARGET_CHANNELS:
            raise ValueError(f"{src_path}: unexpected channels={f.channels}")
        if f.frames < n_samples:
            raise ValueError(f"{src_path}: only {f.frames} frames, need {n_samples}")
        data = f.read(frames=n_samples, dtype="float32", always_2d=False)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_path), data, TARGET_SR, subtype="PCM_16")
    # validate roundtrip
    info = sf.info(str(dst_path))
    if abs(info.frames - n_samples) > 1:
        raise ValueError(f"{dst_path}: wrote {info.frames}, expected {n_samples}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{dst_path}: NaN/Inf samples")
    return int(info.frames)


def slug(length_sec: float) -> str:
    return f"{int(length_sec)}s"


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("query_variants")

    total_written = 0
    failures = 0

    for length_sec in TARGET_LENGTHS_SEC:
        n_samples = int(round(length_sec * TARGET_SR))
        length_slug = slug(length_sec)
        new_root = DATA / f"queries_{length_slug}"
        log.info(f"=== building {length_slug} variants → {new_root} ===")

        for corpus in CORPORA:
            for manifest_in, manifest_kind, subdir_inside_queries in VARIANTS:
                src_csv = MANIFEST / corpus / manifest_in
                if not src_csv.exists():
                    log.warning(f"skip: {src_csv} not found")
                    continue
                df = pd.read_csv(src_csv)
                if df.empty:
                    log.warning(f"skip: {src_csv} is empty")
                    continue

                ok_rows: list[dict] = []
                t_local_fail = 0
                for _, r in df.iterrows():
                    src_audio = Path(r["audio_path"])
                    # rewrite the audio_path under queries_{N}s/{corpus}/[ablation/]<name>
                    new_dir = new_root / corpus
                    if subdir_inside_queries:
                        new_dir = new_dir / subdir_inside_queries
                    new_audio = new_dir / src_audio.name
                    try:
                        truncate_wav(src_audio, new_audio, n_samples)
                    except Exception as exc:
                        log.error(f"  FAIL {r['query_id']}: {exc}")
                        t_local_fail += 1
                        continue
                    new_row = dict(r)
                    new_row["audio_path"] = str(new_audio)
                    new_row["length_sec"] = length_sec
                    ok_rows.append(new_row)
                    total_written += 1

                if not ok_rows:
                    log.warning(f"  {corpus}/{manifest_kind}: no rows written")
                    failures += t_local_fail
                    continue

                out_csv = MANIFEST / corpus / f"{manifest_kind}_{length_slug}.csv"
                pd.DataFrame(ok_rows).to_csv(out_csv, index=False)
                log.info(f"  {corpus}/{manifest_kind}: wrote {len(ok_rows)} rows → {out_csv} (failures: {t_local_fail})")
                failures += t_local_fail

    log.info(f"DONE: wrote {total_written} truncated WAVs across {len(TARGET_LENGTHS_SEC)} lengths × {len(CORPORA)} corpora × {len(VARIANTS)} variants (total failures: {failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
