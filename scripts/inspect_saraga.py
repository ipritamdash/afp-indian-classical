"""Senior-grade inspection of a downloaded Saraga sub-corpus.

Produces:
  data/inspection/<corpus>_tracks.csv         — one row per track, all metadata
  data/inspection/<corpus>_works.csv          — composition aggregation (Test B feasibility)
  data/inspection/<corpus>_sections.csv       — section annotation coverage (Ablation feasibility)
  data/inspection/<corpus>_report.md          — human-readable summary + go/no-go

No audio is fully decoded; only soundfile header reads (fast).

Usage:
    uv run python scripts/inspect_saraga.py carnatic
    uv run python scripts/inspect_saraga.py hindustani
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import mirdata
import pandas as pd
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_HOME_ROOT = PROJECT_ROOT / "data" / "mirdata"
OUT_DIR = PROJECT_ROOT / "data" / "inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe(getter, default=None):
    try:
        return getter()
    except Exception:
        return default


def _attr(track, name, default=None):
    try:
        v = getattr(track, name)
        return v if v is not None else default
    except Exception:
        return default


def _meta(track, *keys, default=None):
    """Read from track.metadata (dict). Tries multiple key spellings —
    Saraga is inconsistent between Hindustani (raags/taals) and Carnatic (raaga/taala)."""
    m = _attr(track, "metadata", None)
    if isinstance(m, dict):
        for k in keys:
            v = m.get(k)
            if v is not None:
                return v
    # Fall back to attribute access on the track object itself
    for k in keys:
        v = _attr(track, k, None)
        if v is not None:
            return v
    return default


def _norm_str(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, (list, tuple)):
        # mirdata sometimes returns list of dicts like [{"name":"Ahir Bhairav","mbid":"…"}]
        parts = []
        for item in x:
            if isinstance(item, dict):
                parts.append(str(item.get("name") or item.get("title") or item).strip())
            else:
                parts.append(str(item).strip())
        return "; ".join(p for p in parts if p) or None
    if isinstance(x, dict):
        return str(x.get("name") or x.get("title") or x).strip() or None
    return str(x).strip() or None


def _list_of_names(x) -> list[str]:
    """Flatten Saraga's list-of-dicts; for `artists` items the name is nested at
    item['artist']['name'], for others it's item['name']/item['title']."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if isinstance(item, dict):
                # nested artist case: {"artist": {"name": "..."}, ...}
                nested = item.get("artist")
                if isinstance(nested, dict) and (nested.get("name") or nested.get("title")):
                    out.append(str(nested.get("name") or nested.get("title")).strip())
                    continue
                name = item.get("name") or item.get("title") or item.get("common_name") or item.get("mbid")
                if name:
                    out.append(str(name).strip())
            elif item is not None:
                out.append(str(item).strip())
        return [o for o in out if o]
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, dict):
        nested = x.get("artist")
        if isinstance(nested, dict) and (nested.get("name") or nested.get("title")):
            return [str(nested.get("name") or nested.get("title")).strip()]
        name = x.get("name") or x.get("title") or x.get("common_name") or x.get("mbid")
        return [str(name).strip()] if name else []
    return [str(x).strip()]


def _list_of_mbids(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(d.get("mbid")).strip() for d in x if isinstance(d, dict) and d.get("mbid")]
    if isinstance(x, dict) and x.get("mbid"):
        return [str(x["mbid"]).strip()]
    return []


def _audio_header(path: str | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {"audio_exists": False}
    try:
        info = sf.info(path)
        size_bytes = Path(path).stat().st_size
        return {
            "audio_exists": True,
            "audio_path": path,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration_sec": info.duration,
            "format": info.format,
            "subtype": info.subtype,
            "size_bytes": size_bytes,
        }
    except Exception as e:
        return {"audio_exists": True, "audio_path": path, "audio_error": str(e)}


def _section_annotation_path(track) -> str | None:
    """Saraga ships section annotations as JAMS or annotation paths on the track.
    Search common attribute names."""
    for name in (
        "sections_path",
        "sections_path_a",
        "sections_path_p",
        "section_path",
        "annotation_path_sections",
    ):
        p = _attr(track, name, None)
        if p:
            return p
    return None


def _section_load(track) -> list[tuple[float, float, str]]:
    """Try to read sections as (start, end, label). Falls back to empty list."""
    out: list[tuple[float, float, str]] = []
    sec = _attr(track, "sections", None)
    if sec is None:
        return out
    # mirdata annotation objects expose .intervals (Nx2) and .labels
    intervals = _attr(sec, "intervals", None)
    labels = _attr(sec, "labels", None)
    if intervals is not None and labels is not None:
        try:
            for (s, e), lab in zip(intervals, labels):
                out.append((float(s), float(e), str(lab)))
            return out
        except Exception:
            pass
    # Fall back to a raw list[dict]
    if isinstance(sec, list):
        for d in sec:
            if isinstance(d, dict):
                try:
                    out.append((float(d.get("start", 0)), float(d.get("end", 0)), str(d.get("label", ""))))
                except Exception:
                    continue
    return out


def inspect_corpus(corpus: str) -> int:
    if corpus not in ("carnatic", "hindustani"):
        print(f"ERROR: corpus must be carnatic or hindustani, got {corpus!r}")
        return 2

    dataset_name = f"saraga_{corpus}"
    data_home = DATA_HOME_ROOT / dataset_name
    if not data_home.exists():
        print(f"ERROR: {data_home} does not exist. Run download_saraga.py first.")
        return 3

    print(f"[{corpus}] initializing mirdata loader with data_home={data_home}")
    ds = mirdata.initialize(dataset_name, data_home=str(data_home))
    track_ids = ds.track_ids
    print(f"[{corpus}] track_ids count: {len(track_ids)}")

    # Sample a few attribute names from the first track to map what's available.
    if track_ids:
        first = ds.track(track_ids[0])
        attrs = sorted(a for a in dir(first) if not a.startswith("_") and not callable(getattr(first, a, None)))
        print(f"[{corpus}] sample track attributes ({first.track_id}):")
        for a in attrs:
            print(f"    {a}")

    rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    work_to_perfs: dict[str, list[str]] = defaultdict(list)
    workmbid_to_perfs: dict[str, list[str]] = defaultdict(list)
    n_audio_ok = 0
    n_audio_missing = 0
    n_audio_error = 0
    errors: list[str] = []

    for i, tid in enumerate(track_ids):
        try:
            t = ds.track(tid)
        except Exception as e:
            errors.append(f"track init failed for {tid}: {e}")
            continue

        audio_path = _attr(t, "audio_path", None)
        hdr = _audio_header(audio_path)
        if hdr.get("audio_exists") and "audio_error" not in hdr:
            n_audio_ok += 1
        elif not hdr.get("audio_exists"):
            n_audio_missing += 1
        else:
            n_audio_error += 1

        title = _norm_str(_meta(t, "title"))
        # Saraga Hindustani uses {raags, taals, layas}, Carnatic uses {raaga, taala, …}
        artists = _list_of_names(_meta(t, "album_artists", "artists"))
        raagas = _list_of_names(_meta(t, "raags", "raaga", "raagas"))
        taalas = _list_of_names(_meta(t, "taals", "taala", "taalas"))
        forms = _list_of_names(_meta(t, "forms", "form"))
        works = _list_of_names(_meta(t, "works", "work"))
        work_mbids = _list_of_mbids(_meta(t, "works", "work"))
        track_mbid = _norm_str(_meta(t, "mbid"))
        concert = _norm_str(_meta(t, "concert"))

        row = {
            "track_id": tid,
            "title": title,
            "concert": concert,
            "artists": "; ".join(artists) if artists else None,
            "raagas": "; ".join(raagas) if raagas else None,
            "taalas": "; ".join(taalas) if taalas else None,
            "forms": "; ".join(forms) if forms else None,
            "works": "; ".join(works) if works else None,
            "work_mbids": "; ".join(work_mbids) if work_mbids else None,
            "track_mbid": track_mbid,
            **hdr,
        }
        rows.append(row)

        # work -> performances mapping (for cross-version feasibility)
        for w in works:
            wnorm = w.lower().strip()
            if wnorm:
                work_to_perfs[wnorm].append(tid)
        for wmbid in work_mbids:
            workmbid_to_perfs[wmbid].append(tid)

        # section annotations (for alaap ablation feasibility)
        secs = _section_load(t)
        if secs:
            for s, e, lab in secs:
                section_rows.append(
                    {
                        "track_id": tid,
                        "start_sec": s,
                        "end_sec": e,
                        "label": lab,
                        "duration_sec": max(0.0, e - s),
                    }
                )

        if (i + 1) % 50 == 0:
            print(f"[{corpus}] processed {i+1}/{len(track_ids)} tracks…")

    # ── Save tables ────────────────────────────────────────────────────────
    tracks_df = pd.DataFrame(rows)
    works_rows = []
    for w, perfs in work_to_perfs.items():
        works_rows.append(
            {"work_key": w, "key_type": "title_lower", "n_performances": len(perfs), "tracks": "; ".join(perfs)}
        )
    for wmbid, perfs in workmbid_to_perfs.items():
        works_rows.append(
            {"work_key": wmbid, "key_type": "mbid", "n_performances": len(perfs), "tracks": "; ".join(perfs)}
        )
    works_df = pd.DataFrame(works_rows).sort_values("n_performances", ascending=False) if works_rows else pd.DataFrame()
    sections_df = pd.DataFrame(section_rows)

    tracks_csv = OUT_DIR / f"{corpus}_tracks.csv"
    works_csv = OUT_DIR / f"{corpus}_works.csv"
    sections_csv = OUT_DIR / f"{corpus}_sections.csv"
    tracks_df.to_csv(tracks_csv, index=False)
    if not works_df.empty:
        works_df.to_csv(works_csv, index=False)
    if not sections_df.empty:
        sections_df.to_csv(sections_csv, index=False)

    # ── Aggregate stats for report ─────────────────────────────────────────
    n_tracks = len(tracks_df)
    total_dur = float(tracks_df["duration_sec"].fillna(0).sum()) if "duration_sec" in tracks_df else 0.0
    dur_stats = tracks_df["duration_sec"].describe().to_dict() if "duration_sec" in tracks_df and n_tracks else {}
    sr_counts = Counter(tracks_df["samplerate"].dropna().astype(int).tolist()) if "samplerate" in tracks_df else Counter()
    ch_counts = Counter(tracks_df["channels"].dropna().astype(int).tolist()) if "channels" in tracks_df else Counter()
    fmt_counts = Counter(tracks_df["format"].dropna().tolist()) if "format" in tracks_df else Counter()

    raaga_counts = Counter()
    for r in tracks_df.get("raagas", pd.Series(dtype=str)).dropna():
        for one in str(r).split("; "):
            if one:
                raaga_counts[one] += 1
    taala_counts = Counter()
    for r in tracks_df.get("taalas", pd.Series(dtype=str)).dropna():
        for one in str(r).split("; "):
            if one:
                taala_counts[one] += 1

    n_with_works = int(tracks_df.get("works", pd.Series(dtype=str)).notna().sum())
    n_with_work_mbids = int(tracks_df.get("work_mbids", pd.Series(dtype=str)).notna().sum())

    if not works_df.empty:
        # use mbid keying preferentially, fall back to title-lower
        by_mbid = works_df[works_df.key_type == "mbid"]
        by_title = works_df[works_df.key_type == "title_lower"]
        crossv_mbid = int((by_mbid["n_performances"] >= 2).sum()) if not by_mbid.empty else 0
        crossv_title = int((by_title["n_performances"] >= 2).sum()) if not by_title.empty else 0
        crossv_mbid_3 = int((by_mbid["n_performances"] >= 3).sum()) if not by_mbid.empty else 0
        crossv_title_3 = int((by_title["n_performances"] >= 3).sum()) if not by_title.empty else 0
    else:
        crossv_mbid = crossv_title = crossv_mbid_3 = crossv_title_3 = 0

    n_tracks_with_sections = sections_df["track_id"].nunique() if not sections_df.empty else 0
    section_label_counts = Counter(sections_df["label"].tolist()) if not sections_df.empty else Counter()

    # ── Go/no-go decisions ─────────────────────────────────────────────────
    test_b_ok = (crossv_mbid >= 30) or (crossv_title >= 30)
    ablation_ok = n_tracks_with_sections >= 20  # threshold from roadmap

    # ── Markdown report ────────────────────────────────────────────────────
    md_lines: list[str] = []
    md_lines.append(f"# Saraga {corpus.title()} — Inspection Report\n")
    md_lines.append(f"- Tracks: **{n_tracks}**")
    md_lines.append(f"- Total audio duration: **{total_dur/3600:.1f} h** ({total_dur:.0f} s)")
    if dur_stats:
        md_lines.append(
            f"- Duration per track: min {dur_stats.get('min', 0):.0f}s · "
            f"median {dur_stats.get('50%', 0):.0f}s · mean {dur_stats.get('mean', 0):.0f}s · "
            f"max {dur_stats.get('max', 0):.0f}s"
        )
    md_lines.append(f"- Audio header OK: **{n_audio_ok}**, missing: {n_audio_missing}, decode error: {n_audio_error}")
    md_lines.append(f"- Sample rates: {dict(sr_counts)}")
    md_lines.append(f"- Channels: {dict(ch_counts)}")
    md_lines.append(f"- Formats: {dict(fmt_counts)}")
    md_lines.append("")
    md_lines.append("## Metadata completeness")
    md_lines.append(f"- Tracks with `raagas`: **{int(tracks_df.get('raagas', pd.Series(dtype=str)).notna().sum())} / {n_tracks}**")
    md_lines.append(f"- Tracks with `taalas`: **{int(tracks_df.get('taalas', pd.Series(dtype=str)).notna().sum())} / {n_tracks}**")
    md_lines.append(f"- Tracks with `works`: **{n_with_works} / {n_tracks}**")
    md_lines.append(f"- Tracks with `work_mbids`: **{n_with_work_mbids} / {n_tracks}**")
    md_lines.append(f"- Unique raagas: {len(raaga_counts)}")
    md_lines.append(f"- Unique taalas: {len(taala_counts)}")
    if raaga_counts:
        md_lines.append("- Top raagas: " + ", ".join(f"{n} ({c})" for n, c in raaga_counts.most_common(10)))
    md_lines.append("")
    md_lines.append("## Test B (cross-version) feasibility")
    md_lines.append(f"- Compositions (by MBID) with ≥2 performances: **{crossv_mbid}**")
    md_lines.append(f"- Compositions (by MBID) with ≥3 performances: {crossv_mbid_3}")
    md_lines.append(f"- Compositions (by title-lower) with ≥2 performances: **{crossv_title}**")
    md_lines.append(f"- Compositions (by title-lower) with ≥3 performances: {crossv_title_3}")
    md_lines.append(f"- **Decision (≥30 pairs threshold): {'GO ✅' if test_b_ok else 'NO-GO ❌'}**")
    md_lines.append("")
    md_lines.append("## Ablation (alaap vs composed) feasibility")
    md_lines.append(f"- Tracks with section annotations: **{n_tracks_with_sections}**")
    if section_label_counts:
        md_lines.append("- Section labels (top 15): " + ", ".join(f"{n!r}({c})" for n, c in section_label_counts.most_common(15)))
    md_lines.append(f"- **Decision (≥20 tracks threshold): {'GO ✅' if ablation_ok else 'NO-GO ❌'}**")
    md_lines.append("")
    if errors:
        md_lines.append(f"## Errors during inspection ({len(errors)})")
        for e in errors[:20]:
            md_lines.append(f"- {e}")
        if len(errors) > 20:
            md_lines.append(f"- … and {len(errors)-20} more")
        md_lines.append("")
    md_lines.append("## Output files")
    md_lines.append(f"- `{tracks_csv.relative_to(PROJECT_ROOT)}`")
    if not works_df.empty:
        md_lines.append(f"- `{works_csv.relative_to(PROJECT_ROOT)}`")
    if not sections_df.empty:
        md_lines.append(f"- `{sections_csv.relative_to(PROJECT_ROOT)}`")

    report_path = OUT_DIR / f"{corpus}_report.md"
    report_path.write_text("\n".join(md_lines))

    print(f"\n[{corpus}] WROTE {report_path}")
    print(f"[{corpus}] WROTE {tracks_csv}")
    if not works_df.empty:
        print(f"[{corpus}] WROTE {works_csv}")
    if not sections_df.empty:
        print(f"[{corpus}] WROTE {sections_csv}")
    print(f"[{corpus}] Test B GO/NO-GO: {'GO' if test_b_ok else 'NO-GO'}  (≥2-perf-mbid={crossv_mbid}, ≥2-perf-title={crossv_title})")
    print(f"[{corpus}] Ablation GO/NO-GO: {'GO' if ablation_ok else 'NO-GO'}  (tracks_with_sections={n_tracks_with_sections})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("corpus", choices=["carnatic", "hindustani"])
    args = p.parse_args()
    try:
        return inspect_corpus(args.corpus)
    except Exception:
        traceback.print_exc()
        return 99


if __name__ == "__main__":
    sys.exit(main())
