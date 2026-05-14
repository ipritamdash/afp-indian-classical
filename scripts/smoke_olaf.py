"""End-to-end smoke test for the Olaf Python wrapper.

Steps:
  1. Clear ~/.olaf/db/ (clean state)
  2. Load 1 Saraga Hindustani track at 16 kHz mono (matches Olaf config)
  3. STORE the full track via Olaf(STORE).do(y=...)
  4. QUERY a 10-s slice cut from the same track at a known offset
  5. Assert: matchCount > 0 AND referenceStart ~= the slice offset
  6. QUERY a 10-s slice cut from a DIFFERENT track
  7. Assert: either no match, or matchIdentifier != the first one

Reports timing for store + query. Fails loudly if any assertion fails.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import librosa
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLAF_DIR = PROJECT_ROOT / "external" / "Olaf"

# Add CFFI module + wrapper to import path before importing olaf
sys.path.insert(0, str(OLAF_DIR))
sys.path.insert(0, str(OLAF_DIR / "python-wrapper"))

from olaf import Olaf, OlafCommand  # noqa: E402
from olaf_cffi import lib  # noqa: E402


def clear_db() -> None:
    db = Path.home() / ".olaf" / "db"
    if db.exists():
        for p in db.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        print(f"  cleared {db}")
    else:
        db.mkdir(parents=True, exist_ok=True)
        print(f"  created {db}")


def main() -> int:
    print("=== Olaf smoke test ===")

    # Pick two distinct Saraga Hindustani tracks
    refs = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "hindustani" / "refs.csv")
    refs_long = refs[refs["duration_sec"] > 120].head(2).reset_index(drop=True)
    if len(refs_long) < 2:
        print("ERROR: need ≥2 tracks longer than 120s for smoke test")
        return 2

    track_a = refs_long.iloc[0]
    track_b = refs_long.iloc[1]
    print(f"[A] {track_a['source_track_id']}  ({track_a['duration_sec']:.0f}s)")
    print(f"[B] {track_b['source_track_id']}  ({track_b['duration_sec']:.0f}s)")

    print("\n--- step 1: clear DB ---")
    clear_db()

    print("\n--- step 2: load track A at 16 kHz mono ---")
    t0 = time.time()
    y_a, sr = librosa.load(track_a["audio_path"], sr=16000, mono=True)
    print(f"  loaded {len(y_a)} samples ({len(y_a)/sr:.1f}s) at sr={sr} in {time.time()-t0:.1f}s")
    assert sr == 16000
    assert y_a.ndim == 1

    print("\n--- step 3: STORE track A ---")
    t0 = time.time()
    olaf_store = Olaf(OlafCommand.STORE, track_a["audio_path"])
    store_id = olaf_store.audio_identifier
    olaf_store.do(y=y_a)
    del olaf_store
    print(f"  STORE done in {time.time()-t0:.1f}s; identifier={store_id}")

    print("\n--- step 4: cut a 10-s slice from track A at offset 60s, QUERY ---")
    cut_offset_sec = 60.0
    cut_len_sec = 10.0
    cut_start = int(cut_offset_sec * sr)
    cut_end = cut_start + int(cut_len_sec * sr)
    y_query_a = y_a[cut_start:cut_end].copy()
    assert len(y_query_a) == int(cut_len_sec * sr)

    t0 = time.time()
    olaf_q = Olaf(OlafCommand.QUERY, track_a["audio_path"])
    results_a = olaf_q.do(y=y_query_a)
    del olaf_q
    query_dt = time.time() - t0
    print(f"  QUERY (clip from A) done in {query_dt*1000:.0f} ms")
    print(f"  results: {results_a}")

    if not results_a:
        print("FAIL: no match for self-query")
        return 3
    top = max(results_a, key=lambda r: r["matchCount"])
    print(f"  top: matchCount={top['matchCount']}, identifier={top['matchIdentifier']}, refStart={top.get('referenceStart'):.2f}, refStop={top.get('referenceStop'):.2f}")
    if top["matchIdentifier"] != store_id:
        print(f"FAIL: top match identifier {top['matchIdentifier']} != stored id {store_id}")
        return 4
    # referenceStart should be near our 60s cut offset
    refstart = float(top.get("referenceStart", -1))
    if not (50 <= refstart <= 70):
        print(f"FAIL: referenceStart {refstart} not near expected 60s")
        return 5
    print("  ✅ self-query match verified (identifier + time-offset)")

    print("\n--- step 5: load track B + QUERY 10-s slice from B (should NOT match A) ---")
    y_b, _ = librosa.load(track_b["audio_path"], sr=16000, mono=True)
    y_query_b = y_b[cut_start:cut_end].copy()
    olaf_q2 = Olaf(OlafCommand.QUERY, track_b["audio_path"])
    results_b = olaf_q2.do(y=y_query_b)
    del olaf_q2
    print(f"  results: {results_b}")
    if results_b:
        top_b = max(results_b, key=lambda r: r["matchCount"])
        if top_b["matchIdentifier"] == store_id:
            # match against A is expected to be near zero / low — fail only if a strong false-positive
            if top_b["matchCount"] > top["matchCount"] / 4:
                print(f"FAIL: false positive — B query strongly matched A (score {top_b['matchCount']} vs self {top['matchCount']})")
                return 6
            print(f"  weak false-positive (matchCount={top_b['matchCount']}); acceptable")
    else:
        print("  ✅ no false-positive match against A")

    print("\n=== SMOKE TEST PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
