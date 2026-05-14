"""Inspect the Dejavu DB state while indexing is still in flight.

Read-only — does not touch the writer's connections. Sanity-checks:
  1. All songs marked fingerprinted=1 (no half-inserted rows)
  2. Hash counts make sense vs track duration (rough rate ~2-3k hashes/min)
  3. Distribution: min/median/max hashes per track; any outliers?
  4. Coverage: how many Hindustani vs Carnatic done?
  5. Sample spot-check: pick a query whose source IS indexed, run it through
     find_matches + align_matches, confirm rank-1 is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "external" / "dejavu"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "systems"))

# Lazy import of Dejavu only if needed for the spot check
PG_HOST = "/Users/prita/.afp_pgserver"
DB_NAME = "dejavu_saraga"


def read_songs() -> pd.DataFrame:
    """Pull songs table as pandas DF. Use a fresh read-only connection."""
    conn = psycopg2.connect(host=PG_HOST, dbname=DB_NAME, user="postgres")
    df = pd.read_sql_query(
        'SELECT song_id, song_name, fingerprinted, total_hashes, date_created '
        'FROM songs ORDER BY date_created',
        conn,
    )
    conn.close()
    return df


def read_refs() -> pd.DataFrame:
    h = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "hindustani" / "refs.csv")
    c = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "carnatic" / "refs.csv")
    return pd.concat([h, c], ignore_index=True)


def main() -> None:
    print("=== Songs indexed so far ===")
    songs = read_songs()
    print(f"total rows in songs table: {len(songs)}")
    n_fp = (songs["fingerprinted"] == 1).sum()
    n_unfp = (songs["fingerprinted"] != 1).sum()
    print(f"  fingerprinted=1: {n_fp}")
    print(f"  fingerprinted=0 (in flight / failed): {n_unfp}")

    if n_fp == 0:
        print("nothing fingerprinted yet — exiting")
        return

    fp = songs[songs["fingerprinted"] == 1].copy()
    print()
    print("=== hash count distribution ===")
    h = fp["total_hashes"]
    print(f"  min:    {h.min():,}")
    print(f"  median: {h.median():,.0f}")
    print(f"  mean:   {h.mean():,.0f}")
    print(f"  max:    {h.max():,}")
    print(f"  total:  {h.sum():,}")

    # Outliers (more than 3 std from mean, or zero hashes)
    z_low = fp[h == 0]
    if not z_low.empty:
        print(f"  ANOMALY: {len(z_low)} songs with 0 hashes:")
        print(z_low[["song_id", "song_name"]].to_string())

    print()
    print("=== coverage by corpus ===")
    fp["corpus"] = fp["song_name"].str.split("_", n=1).str[0]
    cov = fp.groupby("corpus").size()
    print(cov.to_string())

    print()
    print("=== hashes per minute of audio (vs refs.csv) ===")
    refs = read_refs()
    refs["corpus"] = refs["ref_id"].str.split("_", n=1).str[0]
    joined = fp.merge(refs[["ref_id", "duration_sec"]], left_on="song_name", right_on="ref_id", how="left")
    joined["hashes_per_min"] = joined["total_hashes"] / (joined["duration_sec"] / 60.0)
    hpm = joined["hashes_per_min"].dropna()
    print(f"  min:    {hpm.min():.0f}")
    print(f"  median: {hpm.median():.0f}")
    print(f"  max:    {hpm.max():.0f}")
    print(f"  std:    {hpm.std():.0f}")
    # Flag songs with anomalously LOW hashes/min (potential bad indexing)
    cutoff = hpm.quantile(0.05)
    low = joined[joined["hashes_per_min"] < cutoff].sort_values("hashes_per_min")
    if not low.empty:
        print(f"  bottom-5%-tile songs (hpm < {cutoff:.0f}):")
        print(low[["song_name", "duration_sec", "total_hashes", "hashes_per_min"]].head(5).to_string())

    print()
    print("=== top-10 most recently indexed ===")
    print(fp.tail(10)[["song_id", "song_name", "total_hashes", "date_created"]].to_string(index=False))

    print()
    print("=== spot-check: query a clip from an already-indexed source ===")
    import warnings
    warnings.filterwarnings("ignore")
    import dejavu_runner  # noqa: imports trigger Dejavu setup; sets up FastPostgreSQLDatabase
    import dejavu as _djv
    import soundfile as sf
    import librosa
    import numpy as np
    from dejavu.config.settings import (
        HASHES_MATCHED, OFFSET_SECS, SONG_NAME,
    )

    # find a query whose ref_id is among the indexed songs
    indexed_names = set(fp["song_name"].tolist())
    h_q = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "hindustani" / "queries_44k.csv")
    c_q = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "carnatic" / "queries_44k.csv")
    q_all = pd.concat([h_q, c_q], ignore_index=True)
    eligible = q_all[q_all["ref_id"].isin(indexed_names)]
    if eligible.empty:
        print("  no queries map to currently-indexed songs (yet); skipping")
        return
    pick = eligible.sample(1, random_state=42).iloc[0]
    qid, qpath, expected_ref = pick["query_id"], pick["audio_path"], pick["ref_id"]
    expected_offset = pick["offset_sec"]
    print(f"  query: {qid} (source ref: {expected_ref}, expected offset {expected_offset:.2f}s)")

    # Use a fresh Dejavu instance pointing at the already-running pgserver
    djv = dejavu_runner.build_dejavu()
    # Load the query WAV at 44.1 kHz int16 (same path the runner uses)
    with sf.SoundFile(qpath) as f:
        sr = f.samplerate
        data = f.read(dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != 44100:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=44100, res_type="soxr_hq")
    samples = np.clip(mono * 32767.0, -32768.0, 32767.0).astype(np.int16)

    hashes, _ = djv.generate_fingerprints(samples, Fs=44100)
    matches, dedup_hashes, _ = djv.find_matches(hashes)
    aligned = djv.align_matches(matches, dedup_hashes, len(hashes), topn=5)
    if not aligned:
        print("  ❌ no match returned")
        return
    print("  top 5 results:")
    for r, h in enumerate(aligned, 1):
        name = h[SONG_NAME].decode("utf-8") if isinstance(h[SONG_NAME], bytes) else h[SONG_NAME]
        print(f"    rank {r}: {name}  match_count={h[HASHES_MATCHED]:>5}  offset_sec={h[OFFSET_SECS]:.2f}")
    top = aligned[0]
    top_name = top[SONG_NAME].decode("utf-8") if isinstance(top[SONG_NAME], bytes) else top[SONG_NAME]
    correct = top_name == expected_ref
    off_err = abs(top[OFFSET_SECS] - expected_offset)
    print(f"  rank-1 correct: {'✅' if correct else '❌'}  (offset error {off_err:.3f}s)")


if __name__ == "__main__":
    main()
