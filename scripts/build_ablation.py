"""Cut section-aligned 10-s queries for the alaap-vs-composed ablation.

Reads per-corpus section annotations from data/inspection/<corpus>_sections.csv
and source audio paths from data/manifests/<corpus>/refs.csv. For every section
that (a) is at least LONGEST_QUERY_SEC long and (b) classifies as alaap /
composed / tani, cut one 10-s query centred on the section midpoint with a
seeded jitter of ±0.5 s.

Reproducibility contract:
  * Per-section seed derived from SHA-256(GLOBAL_SEED, corpus, track_id, start_sec, end_sec) → 4 bytes
  * All offsets sample-accurate via soundfile partial reads
  * Resampled to 16 kHz mono PCM_16
  * Validated: duration == LONGEST_QUERY_SEC ± 1 sample, no NaN

Outputs:
  data/queries/<corpus>/ablation/<query_id>.wav
  data/manifests/<corpus>/queries_ablation.csv

Usage:
  uv run python scripts/build_ablation.py hindustani
  uv run python scripts/build_ablation.py carnatic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse helpers from build_testset.py (defined as module-level functions)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_testset import (  # noqa: E402
    LONGEST_QUERY_SEC,
    TARGET_SR,
    TARGET_CHANNELS,
    load_segment,
    write_wav,
    validate_wav,
)
from manifest import GLOBAL_SEED  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = PROJECT_ROOT / "data" / "manifests"
INSP_DIR = PROJECT_ROOT / "data" / "inspection"
QUERIES_DIR = PROJECT_ROOT / "data" / "queries"


# Section-type classifiers — every needle is matched case-insensitively against
# the section label AFTER NFKD-normalisation + diacritic-strip (Unicode Mn class).
# Anchored in:
#   * Hindustani: Ālāp-1, Khyāl (…)-N, Tarānā (…)-N — Wikipedia "Alap" + Saraga schema.
#     Plus genuinely-composed Hindustani genres found in Saraga 1.5: Ṭhumri, Bhajan,
#     Dādrā (all are composed lyrics-bearing forms).
#   * Carnatic: Vocal ālāp, Violin ālāp, Pallavi, Anupallavi, Caraṇam, Tani āvartana
#     — fsmbuddy.com / kalasudha.com Carnatic kriti structure refs.
#
# Intentionally NOT bucketed:
#   - Laggī (Hindustani tabla percussion finale; closer to Carnatic tani-āvartana than to
#     composed; ambiguous — kept out of both buckets rather than miscategorised). Documented.
#   - Kalpanā svara, Neraval, Muktāyi svara, Ciṭṭa svara, Jati svara, Tānam (intermediate
#     onset density — would muddy alaap-vs-composed contrast).
#   - Recitation-* (spoken intros), --N (unlabelled).
#
# Needles are ASCII-after-diacritic-strip — same normalisation is applied to the label
# at match time, so 'Ṭhumri'.normalize() == 'thumri' matches the 'thumri' needle.
def _normalize(s: str) -> str:
    """NFKD-decompose, strip combining marks (Unicode category Mn), lowercase.
    'Ṭhumri' → 'thumri';  'Caraṇam' → 'caranam';  'Khyāl' → 'khyal'."""
    nfkd = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return stripped.lower()


ALAAP_NEEDLES = ("alap", "alapana")
COMPOSED_NEEDLES = (
    "pallavi", "anupallavi", "caranam",
    "khyal", "kriti", "kirtana",
    "tarana", "bandish", "thumri",
    "bhajan", "dadra",
)
TANI_NEEDLES = ("tani",)


def classify(label: str) -> str | None:
    if not isinstance(label, str):
        return None
    s = _normalize(label)
    if any(n in s for n in TANI_NEEDLES):
        return "tani"
    if any(n in s for n in ALAAP_NEEDLES):
        return "alaap"
    if any(n in s for n in COMPOSED_NEEDLES):
        return "composed"
    return None


def per_section_seed(corpus: str, track_id: str, start_sec: float, end_sec: float) -> int:
    payload = f"{GLOBAL_SEED}:{corpus}:{track_id}:{start_sec:.3f}:{end_sec:.3f}"
    h = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(h[:4], "big")


def build(corpus: str) -> int:
    if corpus not in ("hindustani", "carnatic"):
        print(f"ERROR corpus must be hindustani|carnatic, got {corpus!r}")
        return 2

    sections_csv = INSP_DIR / f"{corpus}_sections.csv"
    refs_csv = MANIFESTS / corpus / "refs.csv"
    if not sections_csv.exists():
        print(f"ERROR sections csv missing: {sections_csv}")
        return 3
    if not refs_csv.exists():
        print(f"ERROR refs csv missing: {refs_csv}")
        return 3

    out_q_dir = QUERIES_DIR / corpus / "ablation"
    out_q_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = MANIFESTS / corpus / "queries_ablation.csv"

    log_path = MANIFESTS / corpus / "build_ablation.log"
    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    log = logging.getLogger("ablation")

    sec = pd.read_csv(sections_csv)
    refs = pd.read_csv(refs_csv)
    log.info(f"sections: {len(sec)} rows; refs: {len(refs)} rows")

    # join sections → refs on track_id
    refs_min = refs[["source_track_id", "audio_path", "duration_sec",
                     "raagas", "taalas", "artists", "works"]].rename(
        columns={"source_track_id": "track_id", "duration_sec": "track_duration_sec"}
    )
    merged = sec.merge(refs_min, on="track_id", how="inner")
    log.info(f"joined: {len(merged)} section rows have a matching ref")

    merged["section_type"] = merged["label"].apply(classify)
    cats = merged.groupby("section_type", dropna=False).size().to_dict()
    log.info(f"section_type counts (incl. None=other): {cats}")

    merged = merged[merged["section_type"].notna()].copy()
    merged = merged[merged["duration_sec"] >= LONGEST_QUERY_SEC].copy()
    log.info(f"kept {len(merged)} sections (alaap|composed|tani, ≥{LONGEST_QUERY_SEC}s)")

    rows: list[dict] = []
    n_failed = 0
    t0 = time.time()

    # Stable ordering — sort to make iteration deterministic across pandas versions
    merged = merged.sort_values(["track_id", "start_sec"], kind="mergesort").reset_index(drop=True)

    seq_per_track: dict[str, int] = {}
    for _, row in merged.iterrows():
        tid = row["track_id"]
        seq = seq_per_track.get(tid, 0)
        seq_per_track[tid] = seq + 1

        seed = per_section_seed(corpus, tid, float(row["start_sec"]), float(row["end_sec"]))
        rng = random.Random(seed)

        start = float(row["start_sec"])
        end = float(row["end_sec"])
        # also clamp by track duration to avoid reading past EOF if section annotations overshoot
        end = min(end, float(row["track_duration_sec"]))
        # midpoint anchor for a 10-s query INSIDE the section
        latest_offset = end - LONGEST_QUERY_SEC
        if latest_offset < start:
            # Section duration constraint already filtered to ≥10s, but track-end clamp
            # could have brought us under. Skip if so.
            continue
        midpoint = (start + latest_offset) / 2.0
        jitter = rng.uniform(-0.5, 0.5)
        offset = round(max(start, min(latest_offset, midpoint + jitter)), 3)

        qid = f"{corpus}_ab_{tid}_s{seq:03d}_{row['section_type']}"
        wav_path = out_q_dir / f"{qid}.wav"
        try:
            audio = load_segment(row["audio_path"], offset, LONGEST_QUERY_SEC, TARGET_SR, TARGET_CHANNELS)
            write_wav(wav_path, audio, TARGET_SR)
            validate_wav(wav_path, LONGEST_QUERY_SEC, TARGET_SR)
        except Exception as e:
            n_failed += 1
            log.error(f"  cut failed for {qid} (offset={offset:.2f}, audio={row['audio_path']}): {e}")
            continue

        rows.append({
            "query_id": qid,
            "ref_id": f"{corpus}_{tid}",
            "corpus": corpus,
            "audio_path": str(wav_path),
            "offset_sec": offset,
            "length_sec": LONGEST_QUERY_SEC,
            "seed": seed,
            "section_type": row["section_type"],
            "section_label": row["label"],
            "section_start": round(float(row["start_sec"]), 3),
            "section_end": round(float(row["end_sec"]), 3),
            "section_duration": round(float(row["duration_sec"]), 3),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_manifest, index=False)
    log.info(f"WROTE {out_manifest}  ({len(df_out)} rows, {n_failed} failed) in {time.time()-t0:.1f}s")

    if not df_out.empty:
        # quick summary
        by_type = df_out.groupby("section_type").size().to_dict()
        by_type_tracks = df_out.groupby("section_type")["ref_id"].nunique().to_dict()
        log.info(f"queries by section_type: {by_type}")
        log.info(f"tracks contributing by section_type: {by_type_tracks}")

    # persist config alongside
    (MANIFESTS / corpus / "config_ablation.json").write_text(json.dumps({
        "global_seed": GLOBAL_SEED,
        "longest_query_sec": LONGEST_QUERY_SEC,
        "target_sr": TARGET_SR,
        "target_channels": TARGET_CHANNELS,
        "alaap_needles": list(ALAAP_NEEDLES),
        "composed_needles": list(COMPOSED_NEEDLES),
        "tani_needles": list(TANI_NEEDLES),
        "n_queries": len(df_out),
        "n_failed": n_failed,
        "queries_by_section_type": (df_out.groupby("section_type").size().to_dict() if not df_out.empty else {}),
        "tracks_by_section_type": (df_out.groupby("section_type")["ref_id"].nunique().to_dict() if not df_out.empty else {}),
    }, indent=2, ensure_ascii=False))
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
