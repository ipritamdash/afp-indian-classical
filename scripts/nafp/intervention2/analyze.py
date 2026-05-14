"""Intervention 2 — full analysis.

For each of 8 cells × 9 variants:
  - HR@1 + Wilson 95 % CI
  - n_miss, n_miss_same_artist
  - McNemar's exact test (binary outcome paired with baseline α=0)
  - Per pre-registered protocol: Bonferroni / 8 cells = 0.00625

Outputs:
  data/results/nafp/intervention2/analysis/results.csv
  data/results/nafp/intervention2/analysis/SUMMARY.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/nafp/intervention2/analysis"
OUT.mkdir(parents=True, exist_ok=True)

CELLS = ["main_1s", "main_3s", "main_5s", "main_10s",
         "ablation_1s", "ablation_3s", "ablation_5s", "ablation_10s"]
ALPHAS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CONTROLS = ["shuffled", "isotropic"]
BONFERRONI = 0.05 / 8


def queries_for_cell(cell: str) -> pd.DataFrame:
    """Return concat of H+C query manifests."""
    if cell.startswith("main"):
        length = cell.split("_")[1]
        if length == "10s":
            hq, cq = "queries.csv", "queries.csv"
        else:
            hq, cq = f"queries_{length}.csv", f"queries_{length}.csv"
    elif cell.startswith("ablation"):
        length = cell.split("_")[1]
        if length == "10s":
            hq, cq = "queries_ablation.csv", "queries_ablation.csv"
        else:
            hq, cq = f"queries_ablation_{length}.csv", f"queries_ablation_{length}.csv"
    return pd.concat([
        pd.read_csv(REPO / f"data/manifests/hindustani/{hq}"),
        pd.read_csv(REPO / f"data/manifests/carnatic/{cq}"),
    ], ignore_index=True)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfw = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - halfw), min(1.0, center + halfw))


def mcnemar_exact(baseline_hit: np.ndarray, variant_hit: np.ndarray):
    """McNemar's exact test on paired binary outcomes."""
    b = int(((baseline_hit) & (~variant_hit)).sum())  # baseline-hit, variant-miss
    c = int(((~baseline_hit) & (variant_hit)).sum())  # baseline-miss, variant-hit
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0, "delta_hits": 0}
    p = 2 * stats.binom.cdf(min(b, c), n, 0.5)
    p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": n, "p_value": p, "delta_hits": c - b}


def same_artist(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(set(str(a).split(";")) & set(str(b).split(";")))


def main():
    hi_r = pd.read_csv(REPO / "data/manifests/hindustani/refs.csv")
    ca_r = pd.read_csv(REPO / "data/manifests/carnatic/refs.csv")
    refs = pd.concat([hi_r, ca_r], ignore_index=True)
    ref_to_artist = dict(zip(refs.ref_id, refs.artists))

    rows = []
    for cell in CELLS:
        d = REPO / "data/results/nafp/intervention2" / cell
        if not d.exists():
            print(f"[skip] {cell} (no data)")
            continue
        queries = queries_for_cell(cell)
        truth = dict(zip(queries.query_id, queries.ref_id))
        qids = queries.query_id.tolist()

        # baseline (α=0)
        df0 = pd.read_parquet(d / "query_results_alpha_0.00.parquet")
        t0 = df0[df0["rank"] == 1].set_index("query_id")["predicted_ref_id"]
        base_hit = np.array([truth[q] == t0.get(q) for q in qids])

        variant_names = [f"alpha_{a:.2f}" for a in ALPHAS] + [f"control_{c}" for c in CONTROLS]
        for vname in variant_names:
            df = pd.read_parquet(d / f"query_results_{vname}.parquet")
            tv = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"]
            v_hit = np.array([truth[q] == tv.get(q) for q in qids])

            n = len(qids)
            k = int(v_hit.sum())
            ci_lo, ci_hi = wilson_ci(k, n)

            # same-artist breakdown of misses
            miss_mask = ~v_hit
            n_miss = int(miss_mask.sum())
            n_miss_sa = 0
            for q, h in zip(qids, v_hit):
                if not h:
                    pred = tv.get(q)
                    n_miss_sa += int(same_artist(ref_to_artist.get(truth[q]),
                                                  ref_to_artist.get(pred)))

            mc = mcnemar_exact(base_hit, v_hit)
            rows.append({
                "cell": cell, "variant": vname,
                "n": n, "hits": k, "hr@1": k / n,
                "hr@1_ci_lo": ci_lo, "hr@1_ci_hi": ci_hi,
                "n_miss": n_miss, "n_miss_sa": n_miss_sa,
                "mc_b": mc["b"], "mc_c": mc["c"], "mc_delta_hits": mc["delta_hits"],
                "mc_p": mc["p_value"],
                "bonferroni_sig": mc["p_value"] < BONFERRONI,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "results.csv", index=False)
    print(f"[write] {OUT / 'results.csv'}  ({len(out)} rows)")

    # ── Build SUMMARY.md ─────────────────────────────────────────────────
    lines = []
    lines.append("# Intervention 2 — Per-Artist Mean Subtraction: Results")
    lines.append("")
    lines.append(f"Pre-registered protocol: `data/results/nafp/intervention2/PROTOCOL.md` (locked 2026-05-12).")
    lines.append(f"Significance threshold: Bonferroni-corrected α = 0.05 / 8 cells = **{BONFERRONI:.5f}**.")
    lines.append("")
    lines.append("## Per-cell summary (α = 0.10 headline + controls)")
    lines.append("")
    lines.append("| cell | base HR@1 | α=0.10 HR@1 | α=0.10 Δ | α=0.10 p | shuf Δ | iso Δ | best α | best α Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cell in CELLS:
        sub = out[out.cell == cell]
        if len(sub) == 0:
            continue
        b = sub[sub.variant == "alpha_0.00"].iloc[0]
        v10 = sub[sub.variant == "alpha_0.10"].iloc[0]
        sh = sub[sub.variant == "control_shuffled"].iloc[0]
        iso = sub[sub.variant == "control_isotropic"].iloc[0]
        alpha_rows = sub[sub.variant.str.startswith("alpha_")]
        best = alpha_rows.loc[alpha_rows["hr@1"].idxmax()]
        lines.append(
            f"| {cell} | {b['hr@1']:.4f} | {v10['hr@1']:.4f} | "
            f"{v10['mc_delta_hits']:+d} hits | p={v10['mc_p']:.3f} | "
            f"{sh['mc_delta_hits']:+d} | {iso['mc_delta_hits']:+d} | "
            f"{best['variant'].replace('alpha_','')} | {best['mc_delta_hits']:+d} hits |"
        )

    lines.append("")
    lines.append("## Pre-registered decision rule outcomes")
    lines.append("")
    main_alpha10 = out[(out.variant == "alpha_0.10") & (out.cell.str.startswith("main"))]
    n_ge_zero = int((main_alpha10["mc_delta_hits"] >= 0).sum())
    n_bonf = int(main_alpha10["bonferroni_sig"].sum())
    n_regress_sig = int((main_alpha10["bonferroni_sig"] & (main_alpha10["mc_delta_hits"] < 0)).sum())
    iso_vs_v1 = []
    sh_vs_v1 = []
    for cell in CELLS:
        if not cell.startswith("main"):
            continue
        sub = out[out.cell == cell]
        if len(sub) == 0:
            continue
        v10 = sub[sub.variant == "alpha_0.10"]["mc_delta_hits"].iloc[0]
        iso = sub[sub.variant == "control_isotropic"]["mc_delta_hits"].iloc[0]
        sh = sub[sub.variant == "control_shuffled"]["mc_delta_hits"].iloc[0]
        iso_vs_v1.append((cell, v10, iso, sh))

    lines.append(f"1. ≥ 4 of 4 main cells with ΔHR ≥ 0 @ α=0.10: **{n_ge_zero}/4** {'✓' if n_ge_zero >= 4 else '✗'}")
    lines.append(f"2. ≥ 2 of 4 main cells Bonferroni-significant @ α=0.10: **{n_bonf}/4** {'✓' if n_bonf >= 2 else '✗'}")
    lines.append(f"3. No primary cell with Bonferroni-significant regression: **{n_regress_sig} regressions** {'✓' if n_regress_sig == 0 else '✗'}")
    lines.append(f"4. Shuffled-centroid control substantially smaller than V1: per-cell comparison →")
    for cell, v1, iso, sh in iso_vs_v1:
        verdict = "shuffled < V1" if sh < v1 else ("shuffled = V1" if sh == v1 else "shuffled > V1")
        lines.append(f"   - {cell}: V1 Δ={v1:+d}, shuffled Δ={sh:+d}, isotropic Δ={iso:+d} → {verdict}")
    lines.append("")
    lines.append("## Verdict")
    decision_pass = (n_ge_zero >= 4 and n_bonf >= 2 and n_regress_sig == 0)
    lines.append(f"**Adoption criterion (rules 1–3): {'PASS' if decision_pass else 'FAIL'}**")
    iso_meets_or_beats = sum(1 for _, v1, iso, _ in iso_vs_v1 if iso >= v1)
    lines.append(f"**Mechanism check (rule 4): isotropic ≥ V1 in {iso_meets_or_beats}/4 main cells → "
                 f"{'mechanism IS per-artist' if iso_meets_or_beats <= 1 else 'mechanism is NOT per-artist; generic anti-hub effect'}**")
    lines.append("")
    lines.append("## All variants table")
    lines.append("")
    lines.append("| cell | variant | n | hits | HR@1 | 95% CI | n_miss | n_miss_sa | Δhits | McNemar p | Bonf |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in out.iterrows():
        sig = "✓" if r["bonferroni_sig"] else ""
        lines.append(
            f"| {r['cell']} | {r['variant']} | {r['n']} | {r['hits']} | "
            f"{r['hr@1']:.4f} | [{r['hr@1_ci_lo']:.4f}, {r['hr@1_ci_hi']:.4f}] | "
            f"{r['n_miss']} | {r['n_miss_sa']} | {r['mc_delta_hits']:+d} | "
            f"{r['mc_p']:.4f} | {sig} |"
        )
    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[write] {OUT / 'SUMMARY.md'}")
    print()
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
