"""Re-cut existing query manifests at a different target sample rate.

Reads an existing queries manifest (already populated with deterministic
offsets) and re-extracts each 10-s segment from the source MP3 at a new
sample rate. Useful when a system (e.g. Dejavu) is tuned for 44.1 kHz
while another (Olaf) prefers 16 kHz — we keep the same query offsets but
preserve native-rate spectral content for each system.

The reference manifest (refs.csv) is unchanged — source paths stay the same.
Only the query WAVs are re-cut; a new queries CSV is emitted with the
new audio_path pointing at the new WAVs.

Outputs:
  data/queries{suffix}/<corpus>/<query_id>.wav
  data/manifests/<corpus>/queries{suffix}.csv

Usage:
  uv run python scripts/build_queries_at_sr.py hindustani \\
       --queries-in   data/manifests/hindustani/queries.csv \\
       --target-sr    44100 \\
       --suffix       _44k
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_testset import (  # noqa: E402
    LONGEST_QUERY_SEC,
    TARGET_CHANNELS,
    load_segment,
    write_wav,
    validate_wav,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["hindustani", "carnatic"])
    ap.add_argument("--queries-in", required=True, type=Path,
                    help="Existing queries CSV to re-cut")
    ap.add_argument("--refs", type=Path, default=None,
                    help="refs.csv (defaults to data/manifests/<corpus>/refs.csv)")
    ap.add_argument("--target-sr", type=int, required=True)
    ap.add_argument("--suffix", required=True, help="output suffix, e.g. _44k")
    args = ap.parse_args()

    refs_csv = args.refs or (PROJECT_ROOT / "data" / "manifests" / args.corpus / "refs.csv")
    queries_in = args.queries_in if args.queries_in.is_absolute() else PROJECT_ROOT / args.queries_in
    out_q_dir = PROJECT_ROOT / "data" / f"queries{args.suffix}" / args.corpus
    out_q_dir.mkdir(parents=True, exist_ok=True)

    # mirror the same subdir structure: ablation queries live under .../ablation/
    qfile_name = queries_in.stem  # e.g. "queries" or "queries_ablation"
    out_manifest = queries_in.parent / f"{qfile_name}{args.suffix}.csv"
    log_path = queries_in.parent / f"{qfile_name}{args.suffix}.log"

    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("recut")

    log.info(f"target_sr={args.target_sr}, suffix={args.suffix}")
    log.info(f"queries_in: {queries_in}")
    log.info(f"out_q_dir:  {out_q_dir}")
    log.info(f"out_manifest: {out_manifest}")

    refs = pd.read_csv(refs_csv)
    refs_map = dict(zip(refs["ref_id"], refs["audio_path"]))
    queries = pd.read_csv(queries_in)
    log.info(f"input: {len(queries)} queries, {len(refs)} refs")

    rows = []
    n_failed = 0
    t0 = time.time()
    for i, (_, row) in enumerate(queries.iterrows()):
        qid = row["query_id"]
        ref_id = row["ref_id"]
        offset = float(row["offset_sec"])
        length = float(row["length_sec"])
        src_path = refs_map.get(ref_id)
        if src_path is None:
            log.error(f"  no ref for {qid} (ref_id={ref_id})")
            n_failed += 1
            continue

        wav_path = out_q_dir / f"{qid}.wav"
        try:
            audio = load_segment(src_path, offset, length, args.target_sr, TARGET_CHANNELS)
            write_wav(wav_path, audio, args.target_sr)
            validate_wav(wav_path, length, args.target_sr)
        except Exception as e:
            log.error(f"  cut failed for {qid} (offset={offset}, src={src_path}): {e}")
            n_failed += 1
            continue

        new_row = dict(row)
        new_row["audio_path"] = str(wav_path)
        rows.append(new_row)

        if (i + 1) % 200 == 0:
            log.info(f"  cut {i+1}/{len(queries)}")

    out = pd.DataFrame(rows)
    out.to_csv(out_manifest, index=False)
    log.info(f"WROTE {out_manifest} ({len(out)} rows, {n_failed} failed) in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
