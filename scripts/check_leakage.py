"""Composition-leakage audit between test-source and library tracks.

For each corpus, identify any test-source track T that has a "twin" elsewhere
in the library — i.e., another track R that shares at least one composition
(work) with T, either by MusicBrainz work-id or by lowercased work title.

This matters because when we query a 10-s clip from T, an AFP system could
return R (a different recording of the same composition). In Test A this is
noise/ambiguity. In Test B it is the whole point.

Outputs:
    data/inspection/<corpus>_leakage_pairs.csv     pairs (T, R, kind, key)
    data/inspection/leakage_report.md              human-readable summary
    data/manifests/<corpus>/queries.csv            in-place: appends bool col
                                                   `has_twin_in_library`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = PROJECT_ROOT / "data" / "manifests"
INSP_DIR = PROJECT_ROOT / "data" / "inspection"


def split_keys(s: str | float | None) -> set[str]:
    """Split the '; '-joined string into a set of non-empty stripped tokens."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return set()
    return {part.strip() for part in s.split(";") if part.strip()}


def split_keys_lower(s) -> set[str]:
    return {k.lower() for k in split_keys(s)}


def main() -> int:
    report_lines: list[str] = ["# Composition-leakage Audit\n"]
    INSP_DIR.mkdir(parents=True, exist_ok=True)

    for corpus in ("hindustani", "carnatic"):
        refs_csv = MANIFESTS / corpus / "refs.csv"
        queries_csv = MANIFESTS / corpus / "queries.csv"
        if not refs_csv.exists() or not queries_csv.exists():
            report_lines.append(f"## {corpus.title()}\n- SKIP (manifest missing)\n")
            continue

        refs = pd.read_csv(refs_csv)
        queries = pd.read_csv(queries_csv)

        # Index work keys per track
        mbid_keys: dict[str, set[str]] = {
            r["source_track_id"]: split_keys(r.get("work_mbids"))
            for _, r in refs.iterrows()
        }
        title_keys: dict[str, set[str]] = {
            r["source_track_id"]: split_keys_lower(r.get("works"))
            for _, r in refs.iterrows()
        }

        # Which tracks are test sources? (those that appear as ref_id in queries)
        test_source_tids = set(
            queries["ref_id"].apply(lambda rid: rid.split(f"{corpus}_", 1)[1] if rid.startswith(f"{corpus}_") else rid)
        )

        pairs_rows: list[dict] = []
        twins_per_source: dict[str, set[str]] = {}
        all_tids = list(refs["source_track_id"])

        for t_tid in test_source_tids:
            t_mbids = mbid_keys.get(t_tid, set())
            t_titles = title_keys.get(t_tid, set())
            twins: set[str] = set()
            for r_tid in all_tids:
                if r_tid == t_tid:
                    continue
                r_mbids = mbid_keys.get(r_tid, set())
                r_titles = title_keys.get(r_tid, set())

                mbid_overlap = t_mbids & r_mbids
                title_overlap = t_titles & r_titles

                if mbid_overlap or title_overlap:
                    twins.add(r_tid)
                    for key in mbid_overlap:
                        pairs_rows.append({
                            "test_source_tid": t_tid,
                            "library_tid": r_tid,
                            "kind": "mbid",
                            "key": key,
                            "library_tid_is_test_source": r_tid in test_source_tids,
                        })
                    for key in title_overlap:
                        pairs_rows.append({
                            "test_source_tid": t_tid,
                            "library_tid": r_tid,
                            "kind": "title_lower",
                            "key": key,
                            "library_tid_is_test_source": r_tid in test_source_tids,
                        })
            twins_per_source[t_tid] = twins

        pairs_df = pd.DataFrame(pairs_rows)
        out_pairs = INSP_DIR / f"{corpus}_leakage_pairs.csv"
        pairs_df.to_csv(out_pairs, index=False)

        # Update queries.csv in place: add has_twin_in_library
        def _tid_of(ref_id: str) -> str:
            return ref_id.split(f"{corpus}_", 1)[1] if ref_id.startswith(f"{corpus}_") else ref_id
        queries["has_twin_in_library"] = queries["ref_id"].apply(
            lambda rid: len(twins_per_source.get(_tid_of(rid), set())) > 0
        )
        queries.to_csv(queries_csv, index=False)

        # Stats for report
        n_test = len(test_source_tids)
        n_test_with_twin = sum(1 for t in test_source_tids if twins_per_source.get(t))
        n_queries = len(queries)
        n_queries_with_twin = int(queries["has_twin_in_library"].sum())
        n_pairs = len(pairs_df)
        n_pairs_unique_test = pairs_df["test_source_tid"].nunique() if not pairs_df.empty else 0
        n_pairs_unique_lib = pairs_df["library_tid"].nunique() if not pairs_df.empty else 0

        report_lines.append(f"## {corpus.title()}\n")
        report_lines.append(f"- Test-source tracks: **{n_test}**")
        report_lines.append(f"- Test-source tracks with ≥1 twin in library: **{n_test_with_twin}** ({100*n_test_with_twin/max(1,n_test):.1f}%)")
        report_lines.append(f"- Total queries: **{n_queries}**")
        report_lines.append(f"- Queries with `has_twin_in_library=True`: **{n_queries_with_twin}** ({100*n_queries_with_twin/max(1,n_queries):.1f}%)")
        report_lines.append(f"- Total (test, library) overlap pairs: {n_pairs}  (involving {n_pairs_unique_test} unique test tracks and {n_pairs_unique_lib} unique library tracks)")

        if not pairs_df.empty:
            top = pairs_df.groupby("key").size().sort_values(ascending=False).head(8)
            report_lines.append("- Top shared work keys:")
            for k, c in top.items():
                report_lines.append(f"    - `{k}`: {c} pairs")
        report_lines.append("")

        print(f"[{corpus}] {n_test_with_twin}/{n_test} test-source tracks have a twin → {n_queries_with_twin}/{n_queries} queries flagged")
        print(f"[{corpus}] wrote {out_pairs}")

    (INSP_DIR / "leakage_report.md").write_text("\n".join(report_lines))
    print(f"WROTE {INSP_DIR / 'leakage_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
