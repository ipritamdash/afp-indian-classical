"""Materialize the local benchmark artefacts into a Hugging Face Hub-ready staging dir.

Produces `hf_dataset/` at project root with:
- `data/queries/metadata.parquet` + 1000 main query WAVs (AudioFolder pattern)
- `data/queries_ablation/metadata.parquet` + 624 ablation query WAVs
- `data/refs.parquet` (357 ref tracks; LOCAL audio_path stripped — replaced with source_md5
  so the audio cannot be re-downloaded from this artefact alone, satisfying Saraga's
  CC-BY-NC-SA share-alike + the original distributor's role as authoritative source)
- `data/results/{system}_{main,ablation}.{parquet,json}` for the 3 baselined classical systems
- `data/inspection/*.parquet` (tracks/sections/works/leakage_pairs, both corpora combined)
- `data/configs/*.json` (the seeded test-set generation configs — for full reproducibility)

Audio is hardlinked (not copied) from data/queries/ → hf_dataset/data/queries/.../ to avoid
duplicating ~500 MB on disk. Hardlinks behave as independent files for the upload API; only
inode entries are shared.

Run once, then `push_hf_dataset.py` uploads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "data"
STAGING = PROJECT / "hf_dataset"
CORPORA = ("hindustani", "carnatic")


# ── small helpers ─────────────────────────────────────────────────────────

def hardlink_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink (instant, no disk usage). Fall back to copy across-FS."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_parquet(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    print(f"wrote {dst.relative_to(PROJECT)}  ({len(df)} rows, {dst.stat().st_size:,} B)")


# ── 1) queries (main) — AudioFolder pattern ───────────────────────────────

def build_queries(staging: Path) -> dict[str, int]:
    """Combine hindustani+carnatic queries.csv into a single metadata.parquet whose
    `file_name` column points at the relative path of each WAV under the queries/ root.
    Hardlink each WAV into the staging dir."""
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for corpus in CORPORA:
        csv = DATA / "manifests" / corpus / "queries.csv"
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            src_audio = Path(r["audio_path"])
            rel = f"{corpus}/{src_audio.name}"
            dst_audio = staging / "data" / "queries" / rel
            hardlink_or_copy(src_audio, dst_audio)
            rows.append({
                "file_name": rel,
                "query_id": r["query_id"],
                "ref_id": r["ref_id"],
                "corpus": r["corpus"],
                "offset_sec": float(r["offset_sec"]),
                "length_sec": float(r["length_sec"]),
                "seed": int(r["seed"]),
                "has_twin_in_library": bool(r["has_twin_in_library"]),
                "target_sr": 16000,
                "target_channels": 1,
            })
        counts[corpus] = len(df)
    df_all = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    write_parquet(df_all, staging / "data" / "queries" / "metadata.parquet")
    return counts


# ── 2) queries (ablation) ─────────────────────────────────────────────────

def build_queries_ablation(staging: Path) -> dict[str, int]:
    rows: list[dict] = []
    counts: dict[str, int] = {}
    for corpus in CORPORA:
        csv = DATA / "manifests" / corpus / "queries_ablation.csv"
        df = pd.read_csv(csv)
        for _, r in df.iterrows():
            src_audio = Path(r["audio_path"])
            rel = f"{corpus}/{src_audio.name}"  # flatten — drop "ablation/" subdir
            dst_audio = staging / "data" / "queries_ablation" / rel
            hardlink_or_copy(src_audio, dst_audio)
            rows.append({
                "file_name": rel,
                "query_id": r["query_id"],
                "ref_id": r["ref_id"],
                "corpus": r["corpus"],
                "offset_sec": float(r["offset_sec"]),
                "length_sec": float(r["length_sec"]),
                "seed": int(r["seed"]),
                "section_type": r["section_type"],
                "section_label": r["section_label"],
                "section_start": float(r["section_start"]),
                "section_end": float(r["section_end"]),
                "section_duration": float(r["section_duration"]),
                "target_sr": 16000,
                "target_channels": 1,
            })
        counts[corpus] = len(df)
    df_all = pd.DataFrame(rows).sort_values("query_id").reset_index(drop=True)
    write_parquet(df_all, staging / "data" / "queries_ablation" / "metadata.parquet")
    return counts


# ── 3) refs (audio_path STRIPPED) ────────────────────────────────────────

def build_refs(staging: Path) -> int:
    """Combine hindustani+carnatic refs.csv. STRIP the local audio_path (we do not
    redistribute the source MP3s — Saraga's Zenodo deposit is authoritative). Keep
    metadata + source_md5 so users can re-fetch from the original distributor."""
    pieces = []
    for corpus in CORPORA:
        df = pd.read_csv(DATA / "manifests" / corpus / "refs.csv")
        df = df.drop(columns=["audio_path"])  # critical: do not leak local paths
        pieces.append(df)
    df_all = pd.concat(pieces, ignore_index=True).sort_values("ref_id").reset_index(drop=True)
    write_parquet(df_all, staging / "data" / "refs.parquet")
    return len(df_all)


# ── 4) results (per-system × main/ablation) ──────────────────────────────

def build_results(staging: Path) -> None:
    """For each system × length × split combination, copy query_results.parquet +
    scores.json into hf_dataset/data/results/{system}_{cell}.{parquet,json}.

    v0.6 layout: <system>_<split>_<length>.{parquet,json} where:
      system  ∈ {olaf, dejavu, panako, nafp, nmfp, recipe_v3}
      split   ∈ {main, ablation}
      length  ∈ {1s, 3s, 5s, 10s}

    Adds compared to v0.5:
    - All 4 query lengths exposed (was only 10s)
    - NMFP-ckpt-100 (Araz et al. ISMIR 2025) reference ceiling
    - Recipe v3 pooled-McNemar results (3-seed mean per cell)
    """
    # Tuples: (system_label, source_subdir_relative_to_data/results, slug)
    # Classical 4 systems + NMFP, 4 lengths × main+ablation = 40 entries
    classical_systems = ("olaf", "dejavu", "panako", "nafp")
    length_to_subdir = {  # NAFP/baseline 10s case uses bare "saraga_only_main"/"saraga_only_ablation"
        "1s":  ("saraga_only_main_1s",     "saraga_only_ablation_1s"),
        "3s":  ("saraga_only_main_3s",     "saraga_only_ablation_3s"),
        "5s":  ("saraga_only_main_5s",     "saraga_only_ablation_5s"),
        "10s": ("saraga_only_main",        "saraga_only_ablation"),
    }
    entries = []
    for system in classical_systems:
        for length, (main_sub, abl_sub) in length_to_subdir.items():
            entries.append((system, main_sub, f"{system}_main_{length}"))
            entries.append((system, abl_sub,  f"{system}_ablation_{length}"))

    # NMFP-ckpt-100 (Araz et al. 2025) results: nafp/nmfp_eval/saraga_{main,ablation}_<len>/
    for length in ("1s", "3s", "5s", "10s"):
        entries.append(("nafp/nmfp_eval", f"saraga_main_{length}",     f"nmfp_main_{length}"))
        entries.append(("nafp/nmfp_eval", f"saraga_ablation_{length}", f"nmfp_ablation_{length}"))

    # Recipe v3 — 3 seeds × 8 cells. We expose per-seed scores AND pooled-McNemar CSV.
    for seed in (42, 137, 2026):
        for split in ("main", "ablation"):
            for length in ("1s", "3s", "5s", "10s"):
                entries.append(
                    (f"nafp/recipe_v3_30ep", f"seed{seed}_eval/{split}_{length}",
                     f"recipe_v3_seed{seed}_{split}_{length}"),
                )

    n_written = 0
    for system, subdir, slug in entries:
        src_dir = DATA / "results" / system / subdir
        if not src_dir.exists():
            continue
        # scores.json (always)
        src_js = src_dir / "scores.json"
        if src_js.exists():
            dst_js = staging / "data" / "results" / f"{slug}.scores.json"
            dst_js.parent.mkdir(parents=True, exist_ok=True)
            dst_js.write_text(src_js.read_text())
            n_written += 1
        # query_results.parquet (optional — these are large but useful for re-analysis)
        src_pq = src_dir / "query_results.parquet"
        dst_pq = staging / "data" / "results" / f"{slug}.parquet"
        if src_pq.exists():
            df = pd.read_parquet(src_pq)
            write_parquet(df, dst_pq)
    print(f"  [results] wrote {n_written} scores.json files across systems × cells")

    # Recipe v3 pooled-McNemar summary (the headline statistical test)
    src_pool = DATA / "results" / "nafp" / "recipe_v3_30ep" / "pooled_mcnemar.csv"
    if src_pool.exists():
        dst = staging / "data" / "results" / "recipe_v3_pooled_mcnemar.csv"
        shutil.copy(src_pool, dst)
        print(f"  [results] pooled-McNemar table → {dst.name}")

    # Pre-registered protocols (markdown text, small)
    protocols = {
        "PROTOCOL_recipe_v3":   DATA / "results/nafp/recipe_v3_30ep/PROTOCOL.md",
        "PROTOCOL_intervention2": DATA / "results/nafp/intervention2/PROTOCOL.md",
        "RESULTS_recipe_v3":     DATA / "results/nafp/recipe_v3_30ep/RESULTS.md",
    }
    for slug, src in protocols.items():
        if src.exists():
            (staging / "data" / "results" / f"{slug}.md").write_text(src.read_text())
            print(f"  [results] {slug}.md")


# ── 5) inspection (tracks/sections/works/leakage) ────────────────────────

def build_inspection(staging: Path) -> None:
    """The inspection CSVs (tracks metadata, sections, works keys, leakage pairs)
    are derived from Saraga annotations — keep them for reproducibility, but strip
    local audio_path from `tracks` for the same reason as refs."""
    out_dir = staging / "data" / "inspection"

    # tracks (combine, strip local paths)
    pieces = []
    for corpus in CORPORA:
        df = pd.read_csv(DATA / "inspection" / f"{corpus}_tracks.csv")
        df = df.drop(columns=["audio_path"], errors="ignore")
        pieces.append(df)
    write_parquet(pd.concat(pieces, ignore_index=True), out_dir / "tracks.parquet")

    # sections — already corpus-tagged
    write_parquet(pd.read_csv(DATA / "inspection" / "all_sections.csv"),
                  out_dir / "sections.parquet")

    # works
    write_parquet(pd.read_csv(DATA / "inspection" / "all_works.csv"),
                  out_dir / "works.parquet")

    # leakage pairs (per corpus → combine)
    pieces = []
    for corpus in CORPORA:
        df = pd.read_csv(DATA / "inspection" / f"{corpus}_leakage_pairs.csv")
        df["corpus"] = corpus
        pieces.append(df)
    write_parquet(pd.concat(pieces, ignore_index=True), out_dir / "leakage_pairs.parquet")


# ── 6) configs (per corpus × main/ablation) ──────────────────────────────

def build_configs(staging: Path) -> None:
    out_dir = staging / "data" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for corpus in CORPORA:
        for variant, suffix in [("main", "config.json"), ("ablation", "config_ablation.json")]:
            src = DATA / "manifests" / corpus / suffix
            if src.exists():
                dst = out_dir / f"{corpus}_{variant}.json"
                dst.write_text(src.read_text())
                print(f"wrote {dst.relative_to(PROJECT)}")


# ── 7) artefact manifest (root-level hash of every file we put in) ───────

def build_artefact_manifest(staging: Path) -> None:
    """Sha256 every staged file. Lets a reviewer verify the upload integrity
    after the HF mirror caches it."""
    rows = []
    for p in sorted(staging.rglob("*")):
        if p.is_file() and p.name not in {"ARTEFACTS.parquet"}:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rows.append({
                "path": str(p.relative_to(staging)),
                "size_bytes": p.stat().st_size,
                "sha256": h.hexdigest(),
            })
    df = pd.DataFrame(rows)
    write_parquet(df, staging / "ARTEFACTS.parquet")
    total_mb = df["size_bytes"].sum() / 1e6
    print(f"  → {len(df)} files, total {total_mb:.1f} MB")


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    # Only wipe the regenerated data dirs — preserve hand-written docs at the
    # staging root (README.md, LICENSE, LICENSE-CODE, ATTRIBUTION.md, etc.).
    STAGING.mkdir(parents=True, exist_ok=True)
    for sub in ("data", "ARTEFACTS.parquet"):
        p = STAGING / sub
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    print(f"cleared {STAGING}/data and {STAGING}/ARTEFACTS.parquet; docs preserved")

    print("\n[1/7] queries (main)")
    qc = build_queries(STAGING)
    print(f"  hindustani={qc['hindustani']}  carnatic={qc['carnatic']}")

    print("\n[2/7] queries (ablation)")
    qac = build_queries_ablation(STAGING)
    print(f"  hindustani={qac['hindustani']}  carnatic={qac['carnatic']}")

    print("\n[3/7] refs (audio_path stripped)")
    nr = build_refs(STAGING)
    print(f"  refs={nr}")

    print("\n[4/7] results (per system × main/ablation)")
    build_results(STAGING)

    print("\n[5/7] inspection (tracks/sections/works/leakage)")
    build_inspection(STAGING)

    print("\n[6/7] configs")
    build_configs(STAGING)

    print("\n[7/7] artefact manifest (sha256)")
    build_artefact_manifest(STAGING)

    print(f"\nstaging complete: {STAGING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
