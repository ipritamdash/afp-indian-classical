"""Consolidate inspection across all real-audio Indian corpora.

Reads per-corpus *_tracks.csv, *_works.csv, *_sections.csv produced by
inspect_saraga.py, and emits:

    data/inspection/CONSOLIDATED.md     — human-readable cross-corpus go/no-go
    data/inspection/all_works.csv       — combined work-key counts across corpora
    data/inspection/all_sections.csv    — combined section counts across corpora
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSP_DIR = PROJECT_ROOT / "data" / "inspection"


CORPORA = ["carnatic", "hindustani"]


def load_csv(corpus: str, name: str) -> pd.DataFrame:
    p = INSP_DIR / f"{corpus}_{name}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["corpus"] = corpus
    return df


def main() -> None:
    tracks_all = pd.concat([load_csv(c, "tracks") for c in CORPORA], ignore_index=True)
    works_all = pd.concat([load_csv(c, "works") for c in CORPORA], ignore_index=True)
    sections_all = pd.concat([load_csv(c, "sections") for c in CORPORA], ignore_index=True)

    n_tracks = len(tracks_all)
    total_dur_h = float(tracks_all.get("duration_sec", pd.Series(dtype=float)).fillna(0).sum()) / 3600

    # ── Test B (cross-version) across ALL corpora ──────────────────────────
    # work_key is dataset-local; combining naively isn't meaningful (a "Bhairavi" composition
    # in Carnatic kriti repertoire is NOT the same as a "Bhairavi raag" tag in Hindustani).
    # So we report per-corpus AND a union-style "compositions with ≥2 performances anywhere".
    crossv_rows = []
    for c in CORPORA:
        wc = works_all[(works_all.corpus == c)]
        if wc.empty:
            crossv_rows.append({"corpus": c, "by_mbid_ge2": 0, "by_mbid_ge3": 0, "by_title_ge2": 0, "by_title_ge3": 0})
            continue
        m = wc[wc.key_type == "mbid"]
        t = wc[wc.key_type == "title_lower"]
        crossv_rows.append({
            "corpus": c,
            "by_mbid_ge2": int((m.n_performances >= 2).sum()) if not m.empty else 0,
            "by_mbid_ge3": int((m.n_performances >= 3).sum()) if not m.empty else 0,
            "by_title_ge2": int((t.n_performances >= 2).sum()) if not t.empty else 0,
            "by_title_ge3": int((t.n_performances >= 3).sum()) if not t.empty else 0,
        })
    crossv_df = pd.DataFrame(crossv_rows)
    union_mbid_ge2 = int(crossv_df.by_mbid_ge2.sum())
    union_title_ge2 = int(crossv_df.by_title_ge2.sum())

    # ── Ablation (alaap vs composed) across corpora ────────────────────────
    section_counts = sections_all.groupby(["corpus", "label"]).size().reset_index(name="n").sort_values("n", ascending=False)
    # Alaap-like labels (across Hindustani and Carnatic conventions)
    ALAAP_NEEDLES = ["ālāp", "alap", "alāp", "Ālāp", "ālāpana", "alapana"]
    sections_all["is_alaap"] = sections_all.label.fillna("").apply(
        lambda s: any(n.lower() in s.lower() for n in ALAAP_NEEDLES)
    )
    alaap_sections = sections_all[sections_all.is_alaap]
    n_alaap_segments = len(alaap_sections)
    n_alaap_tracks = alaap_sections.track_id.nunique() if not alaap_sections.empty else 0

    COMPOSED_NEEDLES = ["pallavi", "anupallavi", "caraṇam", "caranam", "khyāl", "khyal", "kriti", "kirtana", "tarānā", "bandish", "thumri", "tarana"]
    sections_all["is_composed"] = sections_all.label.fillna("").apply(
        lambda s: any(n.lower() in s.lower() for n in COMPOSED_NEEDLES)
    )
    composed_sections = sections_all[sections_all.is_composed]
    n_composed_segments = len(composed_sections)
    n_composed_tracks = composed_sections.track_id.nunique() if not composed_sections.empty else 0

    # ── Decisions ──────────────────────────────────────────────────────────
    # Test B threshold: ≥30 pairs across the entire benchmark
    test_b_status = "GO ✅" if union_title_ge2 >= 30 or union_mbid_ge2 >= 30 else (
        "BORDERLINE — frame as case study"
        if (union_title_ge2 + union_mbid_ge2) >= 20
        else "NO-GO ❌"
    )
    ablation_status = "GO ✅" if (n_alaap_tracks >= 20 and n_composed_tracks >= 20) else "NO-GO ❌"

    # ── Write CSVs ─────────────────────────────────────────────────────────
    INSP_DIR.mkdir(parents=True, exist_ok=True)
    works_all.to_csv(INSP_DIR / "all_works.csv", index=False)
    sections_all.to_csv(INSP_DIR / "all_sections.csv", index=False)

    # ── Markdown report ────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Consolidated Inspection — All Real-Audio Indian Corpora\n")
    lines.append(f"Corpora included: **{', '.join(c.title() for c in CORPORA)}**")
    lines.append(f"- Total tracks: **{n_tracks}**")
    lines.append(f"- Total audio: **{total_dur_h:.1f} h**")
    lines.append("")
    lines.append("## Per-corpus counts")
    counts = tracks_all.groupby("corpus").agg(
        n=("track_id" if "track_id" in tracks_all.columns else "audio_path", "size"),
        dur_h=("duration_sec", lambda s: s.fillna(0).sum() / 3600),
    )
    lines.append("```")
    lines.append(counts.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Test B (cross-version) feasibility")
    lines.append("```")
    lines.append(crossv_df.to_string(index=False))
    lines.append("```")
    lines.append(f"- Union by MBID (≥2 perf): **{union_mbid_ge2}**")
    lines.append(f"- Union by title (≥2 perf): **{union_title_ge2}**")
    lines.append(f"- **Decision (≥30 threshold): {test_b_status}**")
    lines.append("- Reminder: MBIDs and titles are corpus-local; a Hindustani 'Bhairavi' is not the same composition as a Carnatic 'Bhairavi'. Treat per-corpus pairs only.")
    lines.append("")
    lines.append("## Ablation (alaap vs composed-section) feasibility")
    lines.append(f"- Alaap-like segments (across corpora): **{n_alaap_segments}** across **{n_alaap_tracks}** tracks")
    lines.append(f"- Composed-section segments (across corpora): **{n_composed_segments}** across **{n_composed_tracks}** tracks")
    lines.append(f"- **Decision (≥20 tracks each): {ablation_status}**")
    lines.append("")
    lines.append("## Top section labels (any corpus)")
    lines.append("```")
    top = section_counts.head(25).to_string(index=False)
    lines.append(top)
    lines.append("```")
    lines.append("")
    lines.append("## Output artifacts")
    lines.append("- `data/inspection/all_works.csv`")
    lines.append("- `data/inspection/all_sections.csv`")
    lines.append("- per-corpus reports: `data/inspection/{carnatic,hindustani}_report.md`")

    out_md = INSP_DIR / "CONSOLIDATED.md"
    out_md.write_text("\n".join(lines))
    print(f"WROTE {out_md}")
    print(f"WROTE {INSP_DIR / 'all_works.csv'}")
    print(f"WROTE {INSP_DIR / 'all_sections.csv'}")
    print("")
    print(f"Test B union (mbid≥2): {union_mbid_ge2} → {test_b_status}")
    print(f"Ablation: alaap={n_alaap_tracks}t/{n_alaap_segments}s composed={n_composed_tracks}t/{n_composed_segments}s → {ablation_status}")


if __name__ == "__main__":
    main()
