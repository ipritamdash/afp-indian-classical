"""Stress-test the deployed HF Space across 10+ scenarios.

Categories tested (all real, all verifiable):
  A. In-library Saraga queries (Hindustani + Carnatic × 1s/3s/5s/10s)
  B. Out-of-library FMA clips (short + medium + long)
  C. Known recipe-v3 failure modes (queries that miss in our main_1s eval)
  D. Edge cases: silent audio, sub-1-second clip
"""
from __future__ import annotations
import io
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from gradio_client import Client, handle_file
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
SPACE = "Tachyeon/afp-indian-classical-demo"


def call(client, audio_path: str) -> dict:
    """Returns {verdict, summary, top1, latency_s}."""
    t0 = time.time()
    verdict, summary, top5 = client.predict(handle_file(audio_path), api_name="/_identify")
    latency = time.time() - t0
    top1 = top5["data"][0] if top5.get("data") else None
    return {"verdict": verdict, "summary": summary, "top1": top1, "latency_s": latency}


def first_line(s: str) -> str:
    """Returns first non-empty line of a markdown string."""
    for ln in s.split("\n"):
        if ln.strip():
            return ln.strip()
    return ""


def classify(verdict: str) -> str:
    """Reads the verdict markdown and returns one of {HIGH, MATCH, NO_MATCH, ERROR, OTHER}."""
    v = verdict.lower()
    if "high confidence" in v: return "HIGH"
    if "no match" in v: return "NO_MATCH"
    if "could not decode" in v or "shorter than" in v: return "REJECTED"
    if "match found" in v: return "MATCH"
    return "OTHER"


def main():
    client = Client(SPACE, verbose=False)
    print(f"[ok] connected to {SPACE}\n")

    results = []

    # ── Category A: In-library queries (varied corpus + length) ──────────
    print("─── A. IN-LIBRARY ─────────────────────────────────")
    in_lib_tests = [
        ("hindustani/queries_1s.csv", "hindustani_t0000_q0", "1s Hindustani — easy"),
        ("hindustani/queries_3s.csv", "hindustani_t0000_q0", "3s Hindustani"),
        ("hindustani/queries_5s.csv", "hindustani_t0000_q0", "5s Hindustani"),
        ("carnatic/queries_1s.csv",    "carnatic_t0000_q0",   "1s Carnatic — easy"),
        ("carnatic/queries_5s.csv",    "carnatic_t0030_q1",   "5s Carnatic"),
    ]
    for manifest, qid, desc in in_lib_tests:
        df = pd.read_csv(REPO / "data/manifests" / manifest)
        row = df[df.query_id == qid].iloc[0]
        r = call(client, row.audio_path)
        cls = classify(r["verdict"])
        matched = row.ref_id in str(r["top1"])
        verdict = "✓ PASS" if cls in ("HIGH", "MATCH") and matched else "✗ FAIL"
        score = r["top1"][1] if r["top1"] else None
        print(f"  [{desc:30s}] {qid:20s} cls={cls:8s} score={score}  {verdict}")
        results.append({"category": "A_in_library", "test": desc, "qid": qid,
                        "truth": row.ref_id, "verdict": cls,
                        "score": score, "matched": matched, "pass": (cls in ("HIGH", "MATCH") and matched),
                        "latency_s": r["latency_s"]})

    # ── Category B: Out-of-library FMA clips ──────────────────────────────
    print("\n─── B. OUT-OF-LIBRARY ──────────────────────────────")
    ool = pd.read_csv(REPO / "data/results/threshold_calibration/ool_scores.csv")
    # Pick low-score, mid-score, high-score OOL probes
    ool_sorted = ool.sort_values("top1_score").reset_index(drop=True)
    probe_idx = [10, len(ool_sorted) // 2, len(ool_sorted) - 5]  # low, mid, near-threshold
    probe_descs = ["LOW-score OOL probe", "MID-score OOL probe", "HIGH-score OOL probe (near threshold)"]

    with zipfile.ZipFile(REPO / "data/fma/fma_medium.zip") as zf:
        for idx, desc in zip(probe_idx, probe_descs):
            probe = ool_sorted.iloc[idx]
            data = zf.read(probe["fma_path"])
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(data); fma_path = tf.name
            try:
                r = call(client, fma_path)
                cls = classify(r["verdict"])
                # Demo full clip is longer than calibration 1-sec, so score will be lower
                pre_score = probe["top1_score"]
                live_score = r["top1"][1] if r["top1"] else None
                # Pass: should classify as NO_MATCH for all 3 (calibration shows even high probes < threshold on full clip)
                ok = cls == "NO_MATCH"
                verdict = "✓ PASS" if ok else "✗ FAIL"
                print(f"  [{desc:42s}] pre_1s={pre_score:.4f} live={live_score:.4f} cls={cls:8s}  {verdict}")
                results.append({"category": "B_ool", "test": desc,
                                "pre_score": pre_score, "live_score": live_score,
                                "verdict": cls, "pass": ok, "latency_s": r["latency_s"]})
            finally:
                os.unlink(fma_path)

    # ── Category C: Known recipe-v3 failure modes ─────────────────────────
    print("\n─── C. KNOWN FAILURE MODE (recipe-v3 main_1s misses) ──")
    # Read recipe_v3 seed42 results, find queries it actually missed
    r3_results = pd.read_parquet(REPO / "data/results/nafp/recipe_v3_30ep/seed42_eval/main_1s/query_results.parquet")
    queries_1s = pd.concat([
        pd.read_csv(REPO / "data/manifests/hindustani/queries_1s.csv"),
        pd.read_csv(REPO / "data/manifests/carnatic/queries_1s.csv"),
    ], ignore_index=True)
    truth = dict(zip(queries_1s.query_id, queries_1s.ref_id))
    top1 = r3_results[r3_results["rank"] == 1].set_index("query_id")["predicted_ref_id"]
    misses_seed42 = [q for q in truth if truth[q] != top1.get(q)][:3]
    for qid in misses_seed42:
        row = queries_1s[queries_1s.query_id == qid].iloc[0]
        r = call(client, row.audio_path)
        cls = classify(r["verdict"])
        matched = row.ref_id in str(r["top1"])
        # Expected behavior: recipe_v3 misses → top-1 score may or may not be above threshold
        # If above threshold → demo says MATCH but with wrong ref (this is a "confidently wrong" failure)
        # If below threshold → demo correctly says NO_MATCH
        # We characterize, don't pass/fail
        score = r["top1"][1] if r["top1"] else None
        print(f"  [recipe-v3 miss      ] {qid:25s} truth={row.ref_id[:30]:30s} cls={cls:8s} score={score} matched={matched}")
        results.append({"category": "C_recipe_v3_miss", "qid": qid, "truth": row.ref_id,
                        "verdict": cls, "matched": matched, "score": score, "latency_s": r["latency_s"]})

    # ── Category D: Edge cases ───────────────────────────────────────────
    print("\n─── D. EDGE CASES ──────────────────────────────────")

    # D1: silent 5-second audio → should produce LOW score, demo should say NO MATCH
    print("  [silent 5s audio]")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, np.zeros(8000 * 5, dtype=np.float32), 8000)
        path = tf.name
    try:
        r = call(client, path)
        cls = classify(r["verdict"])
        score = r["top1"][1] if r["top1"] else None
        print(f"    cls={cls}  score={score}  latency={r['latency_s']:.1f}s")
        results.append({"category": "D_edge", "test": "silent_5s", "verdict": cls,
                        "score": score, "latency_s": r["latency_s"]})
    finally:
        os.unlink(path)

    # D2: 0.5-second audio (below window length) → app should reject
    print("  [too-short 0.5s audio]")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, np.random.randn(4000).astype(np.float32) * 0.1, 8000)
        path = tf.name
    try:
        r = call(client, path)
        cls = classify(r["verdict"])
        ok = cls == "REJECTED"
        print(f"    cls={cls}  latency={r['latency_s']:.1f}s  PASS={ok}")
        results.append({"category": "D_edge", "test": "too_short_0.5s", "verdict": cls,
                        "pass": ok, "latency_s": r["latency_s"]})
    finally:
        os.unlink(path)

    # D3: White noise 5s → OOL, low score
    print("  [white noise 5s]")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        sf.write(tf.name, np.random.randn(8000 * 5).astype(np.float32) * 0.1, 8000)
        path = tf.name
    try:
        r = call(client, path)
        cls = classify(r["verdict"])
        score = r["top1"][1] if r["top1"] else None
        ok = cls == "NO_MATCH"
        print(f"    cls={cls}  score={score}  PASS={ok}")
        results.append({"category": "D_edge", "test": "white_noise_5s", "verdict": cls,
                        "score": score, "pass": ok, "latency_s": r["latency_s"]})
    finally:
        os.unlink(path)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n═" * 50)
    print("FINAL SUMMARY")
    print("═" * 50)
    df_results = pd.DataFrame(results)
    for cat in df_results["category"].unique():
        sub = df_results[df_results["category"] == cat]
        if "pass" in sub.columns:
            pass_count = sub["pass"].fillna(False).sum()
            total = len(sub)
            avg_lat = sub["latency_s"].mean()
            print(f"  {cat:25s}: {pass_count}/{total} pass  avg_latency={avg_lat:.1f}s")
    df_results.to_csv(REPO / "data/results/threshold_calibration/live_demo_tests.csv", index=False)
    print(f"\nDetails: data/results/threshold_calibration/live_demo_tests.csv")


if __name__ == "__main__":
    sys.exit(main())
