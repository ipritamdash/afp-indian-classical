"""Failure-mode breakdown: compare NAFP-ckpt-10 baseline vs NMFP-ckpt-100 on Saraga.

For each system + each cell:
  - HR@1 + Wilson 95% CI
  - n_miss
  - n_miss_same_artist (target metric: 14/17 for NAFP baseline)
  - n_miss_same_raaga
  - n_miss_different_corpus
  - McNemar paired test vs NAFP-ckpt-10 baseline

Plus: tracks whether each NAFP miss is still a miss under NMFP, which queries flipped.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2/n
    c = (p + z**2/(2*n))/denom
    h = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)))/denom
    return (max(0, c-h), min(1, c+h))


def mcnemar(a_hit, b_hit):
    b = int(((a_hit) & (~b_hit)).sum())
    c = int(((~a_hit) & (b_hit)).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    return {"b": b, "c": c, "delta": c - b, "p": p}


def same(a, b):
    if pd.isna(a) or pd.isna(b): return False
    return bool(set(str(a).split(";")) & set(str(b).split(";")))


def analyze(qids, truth, df, ref_to_artist, ref_to_raaga, ref_to_corpus):
    top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
    hit = np.array([truth[q] == top1.get(q) for q in qids])
    n = len(qids); k = int(hit.sum())
    ci_lo, ci_hi = wilson_ci(k, n)
    misses = [q for q, h in zip(qids, hit) if not h]
    n_sa = sum(1 for q in misses
               if same(ref_to_artist.get(truth[q]), ref_to_artist.get(top1.get(q))))
    n_sr = sum(1 for q in misses
               if same(ref_to_raaga.get(truth[q]), ref_to_raaga.get(top1.get(q))))
    n_diff_corp = sum(1 for q in misses
                      if ref_to_corpus.get(truth[q]) != ref_to_corpus.get(top1.get(q)))
    return {
        "n": n, "hits": k, "hr@1": k/n,
        "ci": (ci_lo, ci_hi),
        "n_miss": len(misses),
        "n_miss_same_artist": n_sa,
        "n_miss_same_raaga": n_sr,
        "n_miss_diff_corpus": n_diff_corp,
        "hit_arr": hit, "misses": misses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="main_1s")
    ap.add_argument("--length", default="1s")
    args = ap.parse_args()

    # Load manifests
    hi_q = pd.read_csv(REPO / f"data/manifests/hindustani/queries_{args.length}.csv")
    ca_q = pd.read_csv(REPO / f"data/manifests/carnatic/queries_{args.length}.csv")
    queries = pd.concat([hi_q, ca_q], ignore_index=True)
    truth = dict(zip(queries.query_id, queries.ref_id))
    qids = queries.query_id.tolist()

    hi_r = pd.read_csv(REPO / "data/manifests/hindustani/refs.csv")
    ca_r = pd.read_csv(REPO / "data/manifests/carnatic/refs.csv")
    refs = pd.concat([hi_r, ca_r], ignore_index=True)
    a = dict(zip(refs.ref_id, refs.artists))
    r = dict(zip(refs.ref_id, refs.raagas))
    c = dict(zip(refs.ref_id, refs.corpus))

    # Load results
    nafp_path = REPO / f"data/results/nafp/saraga_only_{args.cell}/query_results.parquet"
    nmfp_path = REPO / f"data/results/nafp/nmfp_eval/saraga_{args.cell}/query_results.parquet"
    if not nafp_path.exists():
        print(f"FAIL: NAFP results missing at {nafp_path}")
        return 1
    if not nmfp_path.exists():
        print(f"WARN: NMFP results missing at {nmfp_path} — run NMFP eval first")
        return 1

    nafp_df = pd.read_parquet(nafp_path)
    nmfp_df = pd.read_parquet(nmfp_path)

    nafp = analyze(qids, truth, nafp_df, a, r, c)
    nmfp = analyze(qids, truth, nmfp_df, a, r, c)
    mc = mcnemar(nafp["hit_arr"], nmfp["hit_arr"])

    print(f"\n{'='*70}\n{args.cell} — NAFP-ckpt-10 vs NMFP-ckpt-100 (Araz 2025)\n{'='*70}")
    print(f"\n{'metric':30s} {'NAFP-ckpt10':>14s} {'NMFP-ckpt100':>14s} {'Δ':>10s}")
    print(f"{'-'*70}")
    print(f"{'HR@1':30s} {nafp['hr@1']:>14.4f} {nmfp['hr@1']:>14.4f} {nmfp['hr@1']-nafp['hr@1']:>+10.4f}")
    print(f"{'  95% CI':30s} [{nafp['ci'][0]:.3f}, {nafp['ci'][1]:.3f}]  [{nmfp['ci'][0]:.3f}, {nmfp['ci'][1]:.3f}]")
    print(f"{'n_hits':30s} {nafp['hits']:>14d} {nmfp['hits']:>14d} {nmfp['hits']-nafp['hits']:>+10d}")
    print(f"{'n_miss':30s} {nafp['n_miss']:>14d} {nmfp['n_miss']:>14d}")
    print(f"{'  same-artist miss':30s} {nafp['n_miss_same_artist']:>14d} {nmfp['n_miss_same_artist']:>14d}")
    print(f"{'  same-raaga miss':30s} {nafp['n_miss_same_raaga']:>14d} {nmfp['n_miss_same_raaga']:>14d}")
    print(f"{'  cross-corpus miss':30s} {nafp['n_miss_diff_corpus']:>14d} {nmfp['n_miss_diff_corpus']:>14d}")
    print(f"\nMcNemar paired (b=NAFP-hit/NMFP-miss, c=NAFP-miss/NMFP-hit):")
    print(f"  b={mc['b']}  c={mc['c']}  Δhits={mc['delta']:+d}  p={mc['p']:.4f}")
    if mc["p"] < 0.05/8:
        print(f"  → SIGNIFICANT at Bonferroni 0.05/8 = 0.00625")
    elif mc["p"] < 0.05:
        print(f"  → significant at unadjusted α=0.05 (NOT Bonferroni)")
    else:
        print(f"  → not significant")

    # Per-query flip analysis
    print(f"\nPer-query flip analysis on the 17 NAFP-baseline miss queries:")
    nafp_misses_set = set(nafp["misses"])
    nmfp_misses_set = set(nmfp["misses"])
    fixed = nafp_misses_set - nmfp_misses_set  # NAFP missed, NMFP got
    broken = nmfp_misses_set - nafp_misses_set  # NMFP missed, NAFP got
    common = nafp_misses_set & nmfp_misses_set  # both missed
    print(f"  Fixed by NMFP: {len(fixed)} queries")
    print(f"  Broken by NMFP (NAFP got these): {len(broken)} queries")
    print(f"  Both miss: {len(common)} queries")

    # Save JSON
    out = REPO / "data/results/nafp/nmfp_eval/comparison" / f"{args.cell}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cell": args.cell,
        "nafp": {k: v for k, v in nafp.items() if k not in ["hit_arr", "misses"]},
        "nmfp": {k: v for k, v in nmfp.items() if k not in ["hit_arr", "misses"]},
        "mcnemar": mc,
        "fixed_by_nmfp": sorted(fixed),
        "broken_by_nmfp": sorted(broken),
    }, indent=2, default=str))
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
