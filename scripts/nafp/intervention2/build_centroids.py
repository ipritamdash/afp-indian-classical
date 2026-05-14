"""Intervention 2 — Stage 1: build LOO per-ref artist centroids.

For each ref r with artist a, computes:
    μ_r = mean over all segments of all refs in artist a EXCLUDING r's own segments

Output:
    centroids_per_ref.npy  shape (357, 128) float32  — row i = LOO centroid for ref index i
    ref_index.json         {ref_id: int}             — canonical ref → row index map
    segment_to_ref_idx.npy shape (690414,) int32     — segment global_idx → ref index
    artist_table.parquet   one row per ref           — ref_id, artist_key, cohort_size
    centroids_meta.json    counts, sanity stats

LOO is mandatory: scoring ref r against its own contributions to the centroid would
artificially lower self-similarity, biasing the α-sweep upward. Standard speaker-
recognition protocol (AS-Norm, Matejka 2017).

Singleton-artist refs (cohort=1) have an empty LOO cohort. We zero their centroid
(operationally: intervention is a no-op on them). Verified: 0 such refs on Saraga.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
NAFP_DIR = REPO / "data/results/nafp/saraga_only_main"
OUT_DIR = REPO / "data/results/nafp/intervention2/centroids"


def normalize_artist_key(s: str) -> str:
    """Canonicalize artist field: strip, lowercase, split on ';', sort joined.
    Saraga has 0 multi-artist refs (verified) so this is mostly identity, but
    we keep it safe for any future multi-artist row."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "__UNKNOWN__"
    parts = sorted({p.strip().lower() for p in str(s).split(";") if p.strip()})
    return ";".join(parts) if parts else "__UNKNOWN__"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load lookup + refs ───────────────────────────────────────────────
    lookup = pd.read_parquet(NAFP_DIR / "ref_segment_lookup.parquet")
    refs_hi = pd.read_csv(REPO / "data/manifests/hindustani/refs.csv")
    refs_ca = pd.read_csv(REPO / "data/manifests/carnatic/refs.csv")
    refs = pd.concat([refs_hi, refs_ca], ignore_index=True)
    assert len(refs) == 357, f"expected 357 refs, got {len(refs)}"

    # Build canonical ref_id → ref_idx (0..356)
    unique_refs = lookup["ref_id"].drop_duplicates().tolist()
    assert len(unique_refs) == 357, f"lookup has {len(unique_refs)} unique refs"
    ref_to_idx = {r: i for i, r in enumerate(unique_refs)}
    idx_to_ref = {i: r for r, i in ref_to_idx.items()}

    # Artist per ref (canonicalized)
    refs["artist_key"] = refs["artists"].map(normalize_artist_key)
    refs_by_id = refs.set_index("ref_id")
    artist_per_ref = np.array(
        [normalize_artist_key(refs_by_id.loc[idx_to_ref[i], "artists"]) for i in range(357)],
        dtype=object,
    )

    # Cohort size per artist
    from collections import Counter
    cohort_count = Counter(artist_per_ref.tolist())
    print(f"[info] {len(cohort_count)} unique artists across {len(artist_per_ref)} refs")
    print(f"[info] singleton-artist refs: "
          f"{sum(1 for a in artist_per_ref if cohort_count[a] == 1)}")
    print(f"[info] artists with cohort ≤ 2: "
          f"{sum(1 for a, c in cohort_count.items() if c <= 2)}")
    print(f"[info] artists with cohort ≤ 4: "
          f"{sum(1 for a, c in cohort_count.items() if c <= 4)}")
    print(f"[info] top-5 artists by cohort:")
    for a, c in sorted(cohort_count.items(), key=lambda x: -x[1])[:5]:
        print(f"        {a!r:48s}  cohort={c}")

    # ── Build segment → ref_idx map ──────────────────────────────────────
    lookup_sorted = lookup.sort_values("global_idx").reset_index(drop=True)
    assert (lookup_sorted["global_idx"].values == np.arange(len(lookup_sorted))).all(), \
        "global_idx must be 0..N-1 contiguous"
    seg_to_ref_idx = np.array(
        [ref_to_idx[r] for r in lookup_sorted["ref_id"].values], dtype=np.int32,
    )
    assert seg_to_ref_idx.shape == (690414,), seg_to_ref_idx.shape

    # ── Load ref embeddings (float32 memmap) ─────────────────────────────
    embs_path = NAFP_DIR / "ref_embs.mm"
    embs = np.memmap(embs_path, dtype=np.float32, mode="r", shape=(690414, 128))

    # Sanity: are embeddings already L2-normalized?
    norms = np.linalg.norm(embs[:2000], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"ref embeddings expected L2-normalized, got norm range [{norms.min():.4f}, {norms.max():.4f}]"
    print(f"[ok] ref embeddings L2-normalized (sampled 2000, all ‖e‖≈1)")

    # ── Sum per artist + count per artist (vectorized) ───────────────────
    artist_keys_unique = sorted(set(artist_per_ref.tolist()))
    artist_to_idx = {a: i for i, a in enumerate(artist_keys_unique)}
    A = len(artist_keys_unique)
    print(f"[info] computing per-artist sums over {len(seg_to_ref_idx):,} segments → {A} artists")

    artist_idx_per_seg = np.array(
        [artist_to_idx[artist_per_ref[r]] for r in seg_to_ref_idx], dtype=np.int32,
    )

    sums = np.zeros((A, 128), dtype=np.float64)  # float64 to avoid roundoff over 690k sums
    counts = np.zeros(A, dtype=np.int64)
    # Stream in chunks to bound RAM
    CHUNK = 50_000
    for s in range(0, len(embs), CHUNK):
        e = min(s + CHUNK, len(embs))
        chunk = np.asarray(embs[s:e], dtype=np.float64)
        ai = artist_idx_per_seg[s:e]
        np.add.at(sums, ai, chunk)
        np.add.at(counts, ai, 1)
        sys.stdout.write(f"\r[chunk] {e:,} / {len(embs):,}")
        sys.stdout.flush()
    print()

    # ── Per-ref LOO centroid ─────────────────────────────────────────────
    # μ_r = (artist_sum - sum_of_ref_r_segments) / (artist_count - ref_r_segcount)
    print("[info] computing per-ref LOO centroids (357 refs)")
    centroids = np.zeros((357, 128), dtype=np.float32)
    cohort_sizes = np.zeros(357, dtype=np.int32)
    seg_counts_per_ref = np.bincount(seg_to_ref_idx, minlength=357)

    for r in range(357):
        a = artist_to_idx[artist_per_ref[r]]
        # ref r's own segments
        mask = seg_to_ref_idx == r
        ref_segs = np.asarray(embs[mask], dtype=np.float64)
        ref_sum = ref_segs.sum(axis=0)
        ref_count = ref_segs.shape[0]

        loo_sum = sums[a] - ref_sum
        loo_count = counts[a] - ref_count
        cohort_sizes[r] = (counts[a] // 1) - 0  # in segments; we want #refs though
        if loo_count > 0:
            centroids[r] = (loo_sum / loo_count).astype(np.float32)
        else:
            centroids[r] = np.zeros(128, dtype=np.float32)

    # Cohort size in REFS not segments (more interpretable)
    cohort_size_refs = np.array(
        [cohort_count[artist_per_ref[r]] for r in range(357)], dtype=np.int32,
    )

    # ── Sanity checks ────────────────────────────────────────────────────
    centroid_norms = np.linalg.norm(centroids, axis=1)
    print(f"[stat] centroid norm: min={centroid_norms.min():.4f}  median={np.median(centroid_norms):.4f}  max={centroid_norms.max():.4f}")
    print(f"[stat] cohort-size (refs): min={cohort_size_refs.min()}  median={int(np.median(cohort_size_refs))}  max={cohort_size_refs.max()}")
    print(f"[stat] {(cohort_size_refs == 1).sum()} singleton-artist refs → centroid is zero")

    # Spot-check: for an artist with multiple refs, two different LOO centroids
    # should differ (because we excluded different refs)
    big_artist = max(cohort_count.items(), key=lambda x: x[1])[0]
    big_refs = [r for r in range(357) if artist_per_ref[r] == big_artist][:2]
    diff = np.linalg.norm(centroids[big_refs[0]] - centroids[big_refs[1]])
    print(f"[check] LOO centroids for two refs of artist {big_artist!r}: ‖μ_0 - μ_1‖ = {diff:.4f}")
    assert diff > 1e-4, "LOO centroids identical for two refs of same artist — LOO failed"

    # ── Write artifacts ──────────────────────────────────────────────────
    np.save(OUT_DIR / "centroids_per_ref.npy", centroids)
    np.save(OUT_DIR / "segment_to_ref_idx.npy", seg_to_ref_idx)
    with open(OUT_DIR / "ref_index.json", "w") as f:
        json.dump({"ref_id_by_idx": [idx_to_ref[i] for i in range(357)]}, f, indent=2)

    artist_table = pd.DataFrame({
        "ref_idx": np.arange(357),
        "ref_id": [idx_to_ref[i] for i in range(357)],
        "artist_key": artist_per_ref,
        "cohort_size_refs": cohort_size_refs,
        "n_segments": seg_counts_per_ref,
        "centroid_norm": centroid_norms,
    })
    artist_table.to_parquet(OUT_DIR / "artist_table.parquet", index=False)

    with open(OUT_DIR / "centroids_meta.json", "w") as f:
        json.dump({
            "n_refs": 357,
            "n_segments": int(len(seg_to_ref_idx)),
            "n_artists": A,
            "embedding_dim": 128,
            "cohort_sizes_refs": {
                "min": int(cohort_size_refs.min()),
                "median": int(np.median(cohort_size_refs)),
                "max": int(cohort_size_refs.max()),
                "n_singletons": int((cohort_size_refs == 1).sum()),
            },
            "centroid_norms": {
                "min": float(centroid_norms.min()),
                "median": float(np.median(centroid_norms)),
                "max": float(centroid_norms.max()),
            },
            "loo_protocol": "centroid_r = mean(segs in artist(r)) EXCLUDING segs of ref r itself",
            "source_embs": str(embs_path.relative_to(REPO)),
            "source_lookup": str((NAFP_DIR / "ref_segment_lookup.parquet").relative_to(REPO)),
        }, f, indent=2)

    print(f"[done] wrote {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
