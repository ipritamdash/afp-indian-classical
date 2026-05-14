"""Empirical post-mortem of baseline NAFP-ckpt-10 vs Recipe v3 (3 seeds × 30 ep × BSZ=320).

Every claim in the output is grounded in the actual query_results parquets +
manifest CSVs. No speculation, no synthesis without data.

For each of the 8 cells (main + ablation × 1/3/5/10 s):
  - Per-query hit/miss for baseline + each recipe_v3 seed
  - Cross-seed agreement (which queries are consistently missed?)
  - Per-miss characterization: same-artist, same-raaga, same-corpus, section_type, offset
  - Diff: baseline-miss-recipe_hit (FIXED), baseline-hit-recipe-miss (BROKEN)

Writes: docs/POST_MORTEM_RECIPE_V3.md
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 137, 2026]
OUT = REPO / "docs/POST_MORTEM_RECIPE_V3.md"


def same_set(a, b) -> bool:
    if pd.isna(a) or pd.isna(b): return False
    return bool(set(str(a).split(";")) & set(str(b).split(";")))


def load_refs() -> dict:
    """ref_id -> {artist, raaga, taala, corpus}"""
    hi = pd.read_csv(REPO / "data/manifests/hindustani/refs.csv")
    ca = pd.read_csv(REPO / "data/manifests/carnatic/refs.csv")
    refs = pd.concat([hi, ca], ignore_index=True)
    return {r.ref_id: {"artist": r.artists, "raaga": r.raagas, "taala": r.taalas,
                       "corpus": r.corpus} for _, r in refs.iterrows()}


def load_queries(cell: str) -> pd.DataFrame:
    """Load query manifest with truth + offset + section_type."""
    length = cell.split("_")[1]
    is_abl = "ablation" in cell
    base = "queries_ablation" if is_abl else "queries"
    if length == "10s":
        hq = f"{base}.csv"; cq = f"{base}.csv"
    else:
        hq = f"{base}_{length}.csv"; cq = f"{base}_{length}.csv"
    hi = pd.read_csv(REPO / "data/manifests/hindustani" / hq)
    ca = pd.read_csv(REPO / "data/manifests/carnatic" / cq)
    return pd.concat([hi, ca], ignore_index=True)


def baseline_path(cell: str) -> Path:
    map_ = {
        "main_1s": "saraga_only_main_1s", "main_3s": "saraga_only_main_3s",
        "main_5s": "saraga_only_main_5s", "main_10s": "saraga_only_main",
        "ablation_1s": "saraga_only_ablation_1s", "ablation_3s": "saraga_only_ablation_3s",
        "ablation_5s": "saraga_only_ablation_5s", "ablation_10s": "saraga_only_ablation",
    }
    return REPO / f"data/results/nafp/{map_[cell]}/query_results.parquet"


def recipe_path(cell: str, seed: int) -> Path:
    return REPO / f"data/results/nafp/recipe_v3_30ep/seed{seed}_eval/{cell}/query_results.parquet"


def hits_for(df: pd.DataFrame, truth: dict) -> dict[str, bool]:
    top1 = df[df["rank"] == 1].set_index("query_id")["predicted_ref_id"].to_dict()
    return {q: (truth[q] == top1.get(q)) for q in truth}


def characterize_misses(miss_qids: list[str], queries: pd.DataFrame, truth: dict,
                        top1_map: dict, ref_info: dict) -> dict:
    """Return counts of miss kinds: same-artist / same-raaga / same-corpus / section_type."""
    q_meta = queries.set_index("query_id").to_dict("index")
    same_artist = same_raaga = same_corpus = 0
    section_types = Counter()
    cross_corpus_misses = []
    for q in miss_qids:
        truth_ref = truth[q]
        pred_ref = top1_map.get(q)
        if pred_ref is None: continue
        t_info = ref_info.get(truth_ref, {})
        p_info = ref_info.get(pred_ref, {})
        if same_set(t_info.get("artist"), p_info.get("artist")): same_artist += 1
        if same_set(t_info.get("raaga"), p_info.get("raaga")): same_raaga += 1
        if t_info.get("corpus") == p_info.get("corpus"): same_corpus += 1
        else: cross_corpus_misses.append((q, t_info.get("corpus"), p_info.get("corpus")))
        st = q_meta.get(q, {}).get("section_type", None)
        if st and not (isinstance(st, float) and np.isnan(st)):
            section_types[st] += 1
    return {
        "n_miss": len(miss_qids),
        "same_artist": same_artist,
        "same_raaga": same_raaga,
        "same_corpus": same_corpus,
        "cross_corpus": len(miss_qids) - same_corpus,
        "section_types": dict(section_types),
        "cross_corpus_examples": cross_corpus_misses[:5],
    }


def analyze_cell(cell: str, ref_info: dict) -> dict:
    queries = load_queries(cell)
    truth = dict(zip(queries.query_id, queries.ref_id))
    qids = queries.query_id.tolist()
    has_twin = dict(zip(queries.query_id, queries.has_twin_in_library)) if "has_twin_in_library" in queries.columns else {}

    # Baseline
    bp = baseline_path(cell)
    if not bp.exists():
        return {"cell": cell, "error": f"baseline parquet missing: {bp}"}
    base_df = pd.read_parquet(bp)
    base_top1 = base_df[base_df["rank"] == 1].set_index("query_id")["predicted_ref_id"].to_dict()
    base_hit = {q: truth[q] == base_top1.get(q) for q in qids}
    base_miss = [q for q in qids if not base_hit[q]]

    # Recipe v3 — per seed
    seed_hits = {}
    seed_top1s = {}
    for seed in SEEDS:
        sp = recipe_path(cell, seed)
        if not sp.exists():
            seed_hits[seed] = None; continue
        sdf = pd.read_parquet(sp)
        t1 = sdf[sdf["rank"] == 1].set_index("query_id")["predicted_ref_id"].to_dict()
        seed_top1s[seed] = t1
        seed_hits[seed] = {q: truth[q] == t1.get(q) for q in qids}

    # Cross-seed consensus: a query is "consistently missed" if all 3 seeds miss it
    seed_miss_sets = {s: set(q for q in qids if not seed_hits[s][q]) for s in SEEDS if seed_hits[s]}
    persistent_miss = set.intersection(*seed_miss_sets.values()) if seed_miss_sets else set()
    any_seed_miss = set.union(*seed_miss_sets.values()) if seed_miss_sets else set()
    unanimous_hit = set(qids) - any_seed_miss

    # Per-seed McNemar diffs vs baseline
    per_seed_diffs = {}
    for seed in SEEDS:
        if seed_hits[seed] is None: continue
        sh = seed_hits[seed]
        b = sum(1 for q in qids if base_hit[q] and not sh[q])  # baseline-hit, recipe-miss
        c = sum(1 for q in qids if not base_hit[q] and sh[q])  # baseline-miss, recipe-hit
        per_seed_diffs[seed] = {"b": b, "c": c, "hits": sum(sh.values())}

    # Pooled-McNemar (b+c summed across 3 seeds)
    pooled_b = sum(d["b"] for d in per_seed_diffs.values())
    pooled_c = sum(d["c"] for d in per_seed_diffs.values())
    n_disc = pooled_b + pooled_c
    pooled_p = min(1.0, 2 * stats.binom.cdf(min(pooled_b, pooled_c), n_disc, 0.5)) if n_disc > 0 else 1.0

    # Characterize baseline misses
    base_miss_char = characterize_misses(base_miss, queries, truth, base_top1, ref_info)

    # Characterize recipe_v3 misses (using seed 42 top1 as representative, plus union)
    persistent_misses_list = sorted(persistent_miss)
    if seed_top1s.get(42):
        persistent_char = characterize_misses(persistent_misses_list, queries, truth, seed_top1s[42], ref_info)
    else:
        persistent_char = {}

    # Twin breakdown for baseline misses
    twin_misses = sum(1 for q in base_miss if has_twin.get(q, False))

    # Diff sets
    fixed_qids = sorted([q for q in qids if not base_hit[q] and all(seed_hits[s] and seed_hits[s][q] for s in SEEDS if seed_hits[s])])
    broken_qids = sorted([q for q in qids if base_hit[q] and any(seed_hits[s] and not seed_hits[s][q] for s in SEEDS if seed_hits[s])])
    broken_unanimous = sorted([q for q in qids if base_hit[q] and all(seed_hits[s] and not seed_hits[s][q] for s in SEEDS if seed_hits[s])])

    return {
        "cell": cell,
        "n_queries": len(qids),
        "baseline_hits": sum(base_hit.values()),
        "baseline_miss": len(base_miss),
        "baseline_miss_chars": base_miss_char,
        "baseline_twin_misses": twin_misses,
        "per_seed": per_seed_diffs,
        "pooled_b": pooled_b, "pooled_c": pooled_c, "pooled_p": pooled_p,
        "persistent_miss_n": len(persistent_miss),  # missed by ALL 3 seeds
        "persistent_miss_chars": persistent_char,
        "unanimous_hits": len(unanimous_hit),
        "fixed_by_recipe_n": len(fixed_qids),
        "fixed_sample": fixed_qids[:8],
        "broken_unanimously_n": len(broken_unanimous),
        "broken_unanimous": broken_unanimous,
        "broken_any_seed_n": len(broken_qids),
    }


def write_post_mortem(results: list[dict]):
    out = []
    out.append("# Post-Mortem: Baseline NAFP-ckpt-10 vs Recipe v3 (3 seeds × 30 ep × BSZ=320)\n")
    out.append("**Generated:** 2026-05-14, from query_results.parquet ground truth + Saraga manifests.")
    out.append("**No speculation.** Every number here is computable by re-running `scripts/post_mortem_recipe_v3.py`.\n")
    out.append("## Definitions\n")
    out.append("- **miss** = top-1 predicted ref_id ≠ ground-truth ref_id for that query.")
    out.append("- **same-artist miss** = top-1 prediction is by the same artist as truth (intersect of `artists` string).")
    out.append("- **same-raaga / same-corpus**: analogous.")
    out.append("- **persistent miss** = ALL 3 recipe_v3 seeds (42, 137, 2026) miss this query.")
    out.append("- **fixed by recipe** = baseline misses, ALL 3 seeds hit.")
    out.append("- **broken unanimously** = baseline hits, ALL 3 seeds miss.")
    out.append("- **broken any-seed** = baseline hits, at least 1 seed misses.")
    out.append("- McNemar **b** = baseline-hit & recipe-miss; **c** = baseline-miss & recipe-hit.")
    out.append("- Pooled p = exact 2-sided binomial on min(b+c) across seeds.\n")

    # Per-cell summary table
    out.append("## Per-cell summary\n")
    out.append("| Cell | N | Baseline misses | Recipe v3 mean misses | Pooled b, c | Pooled p | Persistent miss |")
    out.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r.get("error"): continue
        s_misses = [r["per_seed"][s]["hits"] for s in SEEDS if s in r["per_seed"]]
        mean_miss = r["n_queries"] - np.mean(s_misses) if s_misses else None
        out.append(
            f"| {r['cell']} | {r['n_queries']} | {r['baseline_miss']} | "
            f"{mean_miss:.1f} | b={r['pooled_b']}, c={r['pooled_c']} | "
            f"{r['pooled_p']:.2e} | {r['persistent_miss_n']} |"
        )
    out.append("")

    # Section per cell
    for r in results:
        if r.get("error"):
            out.append(f"\n## {r['cell']}: ERROR — {r['error']}\n")
            continue
        c = r["cell"]
        out.append(f"\n## {c}\n")
        out.append(f"- n queries: **{r['n_queries']}**")
        out.append(f"- Baseline hits: **{r['baseline_hits']}** ({r['baseline_hits']/r['n_queries']:.4f})")
        out.append(f"- Baseline misses: **{r['baseline_miss']}**")
        if r["baseline_twin_misses"]:
            out.append(f"- Baseline misses where query has a twin in library: {r['baseline_twin_misses']}")
        seed_hits_str = ", ".join(str(r["per_seed"][s]["hits"]) for s in SEEDS if s in r["per_seed"])
        out.append(f"- Recipe v3 per-seed hits: {seed_hits_str}")
        out.append(f"- Recipe v3 unanimous hits (all 3 seeds): **{r['unanimous_hits']}**")
        out.append(f"- Persistent misses (all 3 seeds miss): **{r['persistent_miss_n']}**")
        out.append(f"- Fixed by recipe (baseline miss → all 3 seeds hit): **{r['fixed_by_recipe_n']}**")
        out.append(f"- Broken any-seed (baseline hit → ≥1 seed miss): {r['broken_any_seed_n']}")
        out.append(f"- Broken unanimously (baseline hit → all 3 miss): {r['broken_unanimously_n']}")
        out.append(f"- Pooled McNemar: b={r['pooled_b']}, c={r['pooled_c']}, p = **{r['pooled_p']:.2e}**")
        out.append("")

        # Baseline miss characterization
        if r["baseline_miss"] > 0:
            bc = r["baseline_miss_chars"]
            out.append(f"### Baseline misses ({r['baseline_miss']}) — what kind of confusions?\n")
            out.append(f"- Same-artist top-1: **{bc['same_artist']} / {bc['n_miss']}** "
                       f"({bc['same_artist']/bc['n_miss']*100:.0f}%)")
            out.append(f"- Same-raaga top-1: {bc['same_raaga']} / {bc['n_miss']}")
            out.append(f"- Same-corpus (both Hindustani or both Carnatic): {bc['same_corpus']} / {bc['n_miss']}")
            out.append(f"- Cross-corpus (HI→CA or CA→HI): {bc['cross_corpus']} / {bc['n_miss']}")
            if bc["section_types"]:
                out.append(f"- Section types of failing queries: {bc['section_types']}")
            if bc["cross_corpus_examples"]:
                out.append("- Cross-corpus examples (query, truth-corpus, pred-corpus):")
                for q, t, p in bc["cross_corpus_examples"]:
                    out.append(f"    - {q}: truth={t}, predicted={p}")
            out.append("")

        # Persistent miss characterization
        if r["persistent_miss_n"] > 0:
            pc = r["persistent_miss_chars"]
            out.append(f"### Persistent misses ({r['persistent_miss_n']}) — what recipe v3 still can't fix\n")
            if pc.get("n_miss"):
                out.append(f"- Same-artist top-1: {pc['same_artist']} / {pc['n_miss']}")
                out.append(f"- Same-raaga top-1: {pc['same_raaga']} / {pc['n_miss']}")
                out.append(f"- Same-corpus: {pc['same_corpus']} / {pc['n_miss']}")
                if pc["section_types"]:
                    out.append(f"- Section types: {pc['section_types']}")
            out.append("")

        # Broken-unanimously analysis
        if r["broken_unanimously_n"] > 0:
            out.append(f"### Broken unanimously ({r['broken_unanimously_n']}) — baseline got these, all 3 recipe seeds lost them\n")
            for q in r["broken_unanimous"][:8]:
                out.append(f"  - {q}")
            if r["broken_unanimously_n"] > 8:
                out.append(f"  - …({r['broken_unanimously_n']-8} more)")
            out.append("")

    return "\n".join(out)


def main():
    cells = ["main_1s", "main_3s", "main_5s", "main_10s",
             "ablation_1s", "ablation_3s", "ablation_5s", "ablation_10s"]
    ref_info = load_refs()
    results = [analyze_cell(c, ref_info) for c in cells]
    md = write_post_mortem(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md)
    print(f"Wrote: {OUT}")
    # Also dump JSON for downstream tooling
    OUT.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
