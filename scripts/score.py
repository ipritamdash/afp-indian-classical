"""Score a system's query_results.parquet against the query manifest.

Reports:
  HR@1, HR@5, HR@10            (top-K hit rate, source-track retrieval)
  MRR@5, MRR@10                 (Mean Reciprocal Rank; standard retrieval metric)
  top1_near                     (top-1 correct AND alignment within ±0.5 s; matches
                                NAFP paper's "top-1 near" — Chang et al. ICASSP 2021)
  top1_near_at_0.05s, _1.0s     (other alignment-tolerance buckets, as rates)
  HR@1_no_twin                  (queries with no composition-twin in library)
  HR@1_with_twin                (queries that DO have a twin)
  offset_err_median_sec         (only on correct top-1 hits)
  offset_err_p95_sec
  Wilson 95% CI on HR@1         (per Wikipedia binomial-proportion CI)
  No-match rate

Usage:
  uv run python scripts/score.py \\
      --results data/results/olaf/saraga_only_main/query_results.parquet \\
      --queries data/manifests/hindustani/queries.csv \\
                data/manifests/carnatic/queries.csv \\
      --output  data/results/olaf/saraga_only_main/scores.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z2 = z * z
    center = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
    radius = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, center - radius), min(1.0, center + radius))


def reciprocal_rank_at_k(merged: pd.DataFrame, k: int, n_queries: int) -> tuple[float, int]:
    """Mean Reciprocal Rank @ K.

    For each query, take 1 / rank_of_first_correct_prediction if that rank is in
    [1, K], else 0. MRR@K = sum(RR) / n_queries (denominator is ALL queries, not
    just those with a hit — standard retrieval convention; see Manning IR ch. 8).

    Returns (mrr_value, n_queries_with_correct_in_topk)."""
    if n_queries == 0:
        return 0.0, 0
    valid = merged[(merged["rank"] >= 1) & (merged["rank"] <= k)]
    correct = valid[valid["predicted_ref_id"] == valid["truth"]]
    if correct.empty:
        return 0.0, 0
    # First correct rank per query
    min_ranks = correct.groupby("query_id")["rank"].min()
    rr_sum = float((1.0 / min_ranks).sum())
    return rr_sum / n_queries, int(len(min_ranks))


def hit_at_k(merged: pd.DataFrame, k: int) -> pd.DataFrame:
    """For each query, was the correct ref_id in the top-K predictions?
    Returns a DataFrame indexed by query_id with columns:
      truth, top_k_preds (list), hit (bool).
    Always returns a DataFrame with a 'hit' column, even when input is empty
    (returns empty DataFrame with that column declared)."""
    correct = []
    # restrict to first K ranks per query (rank 0 = no-match sentinel)
    valid = merged[(merged["rank"] >= 1) & (merged["rank"] <= k)]
    grouped = valid.groupby("query_id")
    for qid, group in grouped:
        truth = group["truth"].iloc[0]
        preds = list(group.sort_values("rank")["predicted_ref_id"])
        correct.append({"query_id": qid, "truth": truth, "top_k_preds": preds, "hit": truth in preds})
    # Also include queries that have no matches at all
    matched_qids = set(grouped.groups.keys())
    all_qids = set(merged["query_id"].unique())
    for qid in all_qids - matched_qids:
        truth = merged.loc[merged.query_id == qid, "truth"].iloc[0]
        correct.append({"query_id": qid, "truth": truth, "top_k_preds": [], "hit": False})
    if not correct:
        # Empty input — return DataFrame with the expected schema so downstream
        # `["hit"].sum()` works without KeyError.
        return pd.DataFrame(columns=["query_id", "truth", "top_k_preds", "hit"])
    return pd.DataFrame(correct)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--queries", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    results = pd.read_parquet(args.results)
    queries = pd.concat([pd.read_csv(p) for p in args.queries], ignore_index=True)
    # Truth ref_id is recorded in the query manifest
    queries["truth"] = queries["ref_id"]

    # Sanity
    assert "query_id" in results.columns, "missing query_id in results"
    assert "rank" in results.columns
    assert "predicted_ref_id" in results.columns
    assert "query_id" in queries.columns

    # Add truth + leakage flag onto results
    cols_keep = ["query_id", "truth"] + [c for c in ("has_twin_in_library", "section_type") if c in queries.columns]
    merged = results.merge(queries[cols_keep], on="query_id", how="left")

    # ── Overall HR@K ───────────────────────────────────────────────────────
    n = queries.shape[0]
    scores: dict[str, object] = {
        "n_queries": int(n),
        "n_results_rows": int(len(results)),
    }
    for k in (1, 5, 10):
        hit_df = hit_at_k(merged, k)
        n_hit = int(hit_df["hit"].sum())
        hr = n_hit / n
        lo, hi = wilson_ci(hr, n)
        scores[f"hr@{k}"] = round(hr, 4)
        scores[f"hr@{k}_ci95"] = [round(lo, 4), round(hi, 4)]
        scores[f"hr@{k}_n_hits"] = n_hit

    # ── MRR@K (Mean Reciprocal Rank) — standard retrieval secondary ──────
    for k in (5, 10):  # MRR@1 == HR@1 by definition, so skip
        mrr, n_with_hit = reciprocal_rank_at_k(merged, k, n)
        scores[f"mrr@{k}"] = round(mrr, 4)
        scores[f"mrr@{k}_n_with_hit"] = n_with_hit

    # ── No-match rate ──────────────────────────────────────────────────────
    # FIXED: use manifest denominator. A query is "no-match" if it has zero
    # rows in results with rank>=1 — this correctly counts queries absent from
    # results.parquet entirely (e.g. crashed worker, silent skip) as no-match,
    # not as zero. Previous bug: groupby on `results` silently dropped queries
    # with no emitted rows from the no_match count.
    qids_with_match = set(results.loc[results["rank"] >= 1, "query_id"].unique())
    qids_all = set(queries["query_id"])
    n_no_match = len(qids_all - qids_with_match)
    scores["n_no_match"] = int(n_no_match)
    scores["frac_no_match"] = round(n_no_match / n, 4)

    # ── Per-corpus breakdown ───────────────────────────────────────────────
    if "corpus" in queries.columns:
        per_corpus = {}
        for corp, qsub in queries.groupby("corpus"):
            mer = merged[merged["query_id"].isin(qsub["query_id"])]
            for k in (1, 5):
                hit_df = hit_at_k(mer, k)
                n_corp = qsub.shape[0]
                n_hit = int(hit_df["hit"].sum())
                hr = n_hit / n_corp
                per_corpus.setdefault(corp, {})[f"hr@{k}"] = round(hr, 4)
                per_corpus[corp][f"hr@{k}_ci95"] = [round(v, 4) for v in wilson_ci(hr, n_corp)]
                per_corpus[corp]["n_queries"] = int(n_corp)
        scores["per_corpus"] = per_corpus

    # ── Twin / no-twin split (only if has_twin_in_library column exists) ───
    if "has_twin_in_library" in queries.columns:
        for label, subset in (
            ("hr@1_no_twin", queries[~queries["has_twin_in_library"]]),
            ("hr@1_with_twin", queries[queries["has_twin_in_library"]]),
        ):
            n_sub = subset.shape[0]
            mer_sub = merged[merged["query_id"].isin(subset["query_id"])]
            hit_df = hit_at_k(mer_sub, 1)
            n_hit = int(hit_df["hit"].sum())
            hr = n_hit / max(1, n_sub)
            scores[label] = round(hr, 4)
            scores[f"{label}_n_queries"] = int(n_sub)
            scores[f"{label}_ci95"] = [round(v, 4) for v in wilson_ci(hr, n_sub)]

    # ── Section-type split for ablation (only if section_type exists) ──────
    if "section_type" in queries.columns:
        per_section = {}
        for stype, qsub in queries.groupby("section_type"):
            n_sub = qsub.shape[0]
            mer_sub = merged[merged["query_id"].isin(qsub["query_id"])]
            for k in (1, 5):
                hit_df = hit_at_k(mer_sub, k)
                n_hit = int(hit_df["hit"].sum())
                hr = n_hit / n_sub
                per_section.setdefault(stype, {})[f"hr@{k}"] = round(hr, 4)
                per_section[stype][f"hr@{k}_ci95"] = [round(v, 4) for v in wilson_ci(hr, n_sub)]
                per_section[stype]["n_queries"] = int(n_sub)
        scores["per_section_type"] = per_section

    # ── Segment-level alignment error (top-1 correct hits only) ──────────
    # Olaf reports ref_start (time in the reference) and query_start (time
    # inside the query at which the match begins). The system's claim is
    # that "query[query_start:] aligns with reference[ref_start:]". The
    # truth is that "query[0:] = source_track[query_offset:]". So the
    # consistent alignment is:
    #     reference[ref_start - query_start :]  ≈  source_track[query_offset :]
    # → alignment error = |(ref_start - query_start) - query_offset|.
    top1 = merged[merged["rank"] == 1].copy()
    top1_correct = top1[top1["predicted_ref_id"] == top1["truth"]].copy()
    q_offsets = queries[["query_id", "offset_sec"]].rename(columns={"offset_sec": "query_offset_sec"})
    top1_correct = top1_correct.merge(q_offsets, on="query_id", how="left")
    if {"ref_start", "query_start"}.issubset(top1_correct.columns):
        top1_correct["align_err_sec"] = (
            (top1_correct["ref_start"] - top1_correct["query_start"]) - top1_correct["query_offset_sec"]
        ).abs()
        errs = top1_correct["align_err_sec"].dropna().values
        scores["align_err_n"] = int(len(errs))
        if len(errs) > 0:
            scores["align_err_median_sec"] = round(float(np.median(errs)), 4)
            scores["align_err_p95_sec"] = round(float(np.percentile(errs, 95)), 4)
            scores["align_err_max_sec"] = round(float(np.max(errs)), 4)
            scores["align_err_within_0.05s"] = int(np.sum(errs <= 0.05))   # ≤ 1 STFT hop @ 8 ms
            scores["align_err_within_0.5s"] = int(np.sum(errs <= 0.5))
            scores["align_err_within_1.0s"] = int(np.sum(errs <= 1.0))
            # ── Top-1-near rates (denominator = n_queries, matches NAFP paper) ──
            # NAFP paper's "top-1 near" uses ±1 segment hop = 0.5s for NAFP (HOP=0.5).
            # We report 3 tolerance buckets as rates over ALL queries (not just
            # top-1-correct ones — that's the convention).
            scores["top1_near_at_0.05s"] = round(int(np.sum(errs <= 0.05)) / n, 4)
            scores["top1_near"]          = round(int(np.sum(errs <= 0.5))  / n, 4)
            scores["top1_near_at_1.0s"]  = round(int(np.sum(errs <= 1.0))  / n, 4)
        else:
            # No top-1 correct hits at all (e.g. Panako on <10s queries).
            # Report zeros explicitly so the field exists for downstream comparison.
            scores["top1_near_at_0.05s"] = 0.0
            scores["top1_near"]          = 0.0
            scores["top1_near_at_1.0s"]  = 0.0
    else:
        # System didn't emit ref_start/query_start — fields absent rather than 0
        scores["top1_near_at_0.05s"] = None
        scores["top1_near"]          = None
        scores["top1_near_at_1.0s"]  = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2, default=str))
    print(json.dumps(scores, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
