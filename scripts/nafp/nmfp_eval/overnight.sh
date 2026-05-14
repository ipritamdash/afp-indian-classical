#!/usr/bin/env bash
# Overnight automation: NMFP × 4 cells + Hubness × 4 cells + comparison + summary.
# Started while NMFP main_1s is already running in background — waits for it
# (poll scores.json), then chains the rest serially.
#
# Each step is logged and recoverable: failure of one step does NOT abort the others.
set -u  # error on undefined var, but DO NOT exit on any non-zero (we want to log + continue)

cd "/Users/prita/Desktop/Audio Fingerprinting/afp_bench"
LOG="data/results/nafp/nmfp_eval/overnight.log"
mkdir -p "$(dirname "$LOG")"
SUMMARY="OVERNIGHT_RESULTS.md"

ts() { date -u +"%Y-%m-%d %H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== OVERNIGHT PIPELINE START ==="
log "Pipeline: NMFP × 4 cells, Hubness × 4 cells, comparison + summary"

# ─── Phase 1: Wait for the already-running NMFP main_1s ──────────────────
# If NMFP hangs / fails, DON'T abort — skip to hubness (which doesn't need NMFP).
NMFP_OK=0
log "Phase 1: waiting for NMFP main_1s (already running in another bg task)"
log "  Polling for scores.json every 30s; max wait 2.5h before falling through"
WAIT_FILE="data/results/nafp/nmfp_eval/saraga_main_1s/scores.json"
WAIT_START=$(date +%s)
while [ ! -f "$WAIT_FILE" ]; do
    sleep 30
    NOW=$(date +%s)
    ELAPSED_MIN=$(( (NOW - WAIT_START) / 60 ))
    if [ "$ELAPSED_MIN" -gt 150 ]; then
        log "  TIMEOUT after 2.5h — NMFP main_1s did not produce scores.json"
        log "  Falling through to Phase 3 (hubness, independent of NMFP)"
        break
    fi
    if [ $(( ELAPSED_MIN % 5 )) -eq 0 ] && [ "$ELAPSED_MIN" -gt 0 ]; then
        log "  ...waiting ($ELAPSED_MIN min elapsed)"
    fi
done
if [ -f "$WAIT_FILE" ]; then
    NMFP_OK=1
    log "Phase 1 DONE: $WAIT_FILE found"
    log "  Result: $(cat "$WAIT_FILE")"
else
    log "Phase 1 SKIPPED — NMFP main_1s did not complete"
fi

# ─── Phase 2: NMFP for the other 3 main cells (re-use index) ─────────────
# Only run if NMFP main_1s succeeded (index file exists)
if [ "$NMFP_OK" -ne 1 ] || [ ! -f "data/results/nafp/nmfp_eval/nmfp_ref_embs.mm" ]; then
    log "Phase 2 SKIPPED — NMFP index unavailable"
else
log "Phase 2: NMFP main_3s, main_5s, main_10s (re-using index)"
for L in 3 5 10; do
    log "  Phase 2.${L}s: NMFP main_${L}s"
    uv run python scripts/nafp/nmfp_eval/run_nmfp_native.py \
        --cell "main_${L}s" \
        --queries "data/manifests/hindustani/queries_${L}s.csv" "data/manifests/carnatic/queries_${L}s.csv" \
        --skip-index >> "$LOG" 2>&1
    if [ -f "data/results/nafp/nmfp_eval/saraga_main_${L}s/scores.json" ]; then
        log "    OK: $(cat data/results/nafp/nmfp_eval/saraga_main_${L}s/scores.json)"
    else
        log "    FAIL: no scores.json produced for main_${L}s"
    fi
done

# Adjust manifest paths for main_10s (manifest file is just "queries.csv", no _10s suffix)
log "  Phase 2.fix: main_10s uses queries.csv (not queries_10s.csv) — re-running if needed"
if [ ! -f "data/results/nafp/nmfp_eval/saraga_main_10s/scores.json" ]; then
    uv run python scripts/nafp/nmfp_eval/run_nmfp_native.py \
        --cell "main_10s" \
        --queries "data/manifests/hindustani/queries.csv" "data/manifests/carnatic/queries.csv" \
        --skip-index >> "$LOG" 2>&1
    log "    main_10s retry: $(cat data/results/nafp/nmfp_eval/saraga_main_10s/scores.json 2>/dev/null || echo 'FAIL')"
fi
fi  # end Phase 2

# ─── Phase 3: Hubness post-processing on existing NAFP ckpt-10 ───────────
log "Phase 3: Hubness post-processing × 4 main cells"
for L in 1 3 5; do
    log "  Phase 3.${L}s: hubness main_${L}s"
    uv run python scripts/nafp/nmfp_eval/hubness_postproc.py \
        --cell "main_${L}s" \
        --queries "data/manifests/hindustani/queries_${L}s.csv" "data/manifests/carnatic/queries_${L}s.csv" \
        --method all >> "$LOG" 2>&1
    if [ -f "data/results/nafp/hubness_postproc/main_${L}s/scores.json" ]; then
        log "    OK: hubness main_${L}s done"
    else
        log "    FAIL: hubness main_${L}s"
    fi
done
# main_10s uses queries.csv
log "  Phase 3.10s: hubness main_10s"
uv run python scripts/nafp/nmfp_eval/hubness_postproc.py \
    --cell "main_10s" \
    --queries "data/manifests/hindustani/queries.csv" "data/manifests/carnatic/queries.csv" \
    --method all >> "$LOG" 2>&1
log "    hubness main_10s status: $([ -f data/results/nafp/hubness_postproc/main_10s/scores.json ] && echo OK || echo FAIL)"

# ─── Phase 4: Failure-mode comparison per cell ───────────────────────────
log "Phase 4: failure-mode comparison NAFP vs NMFP"
for L in 1 3 5 10; do
    log "  Phase 4.${L}s: compare"
    uv run python scripts/nafp/nmfp_eval/compare_failure_modes.py \
        --cell "main_${L}s" --length "${L}s" >> "$LOG" 2>&1
done

# ─── Phase 5: Build OVERNIGHT_RESULTS.md ─────────────────────────────────
log "Phase 5: building $SUMMARY"
uv run python - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path

REPO = Path("/Users/prita/Desktop/Audio Fingerprinting/afp_bench")
out = REPO / "OVERNIGHT_RESULTS.md"

lines = []
lines.append("# Overnight Results — NMFP transfer + Hubness post-processing on Saraga 1.5")
lines.append("")
lines.append("**Goal:** evidence-based gate experiment + alternative attack on the")
lines.append("same-artist failure mode, without retraining.")
lines.append("")
lines.append("**Hypotheses tested:**")
lines.append("- H1 (NMFP transfer): does Araz et al. ISMIR 2025 recipe transfer to Indian classical?")
lines.append("- H2 (hubness): is NAFP's same-artist failure mode a global hub problem that")
lines.append("  inference-time post-processing (Inverted Softmax, CSLS) can attack?")
lines.append("")
lines.append("## Headline numbers (HR@1 on Saraga 1000 main queries)")
lines.append("")
lines.append("| Cell | NAFP-ckpt-10 baseline | NMFP-ckpt-100 (Araz 2025) | NAFP+InvSoftmax | NAFP+CSLS |")
lines.append("|---|---|---|---|---|")
for L in [1, 3, 5, 10]:
    cell = f"main_{L}s"
    base = "0.983 (1s) / 0.998 (3s) / 0.999 (5s) / 1.000 (10s)".split(" / ")[[1,3,5,10].index(L)]
    nmfp_p = REPO / f"data/results/nafp/nmfp_eval/saraga_{cell}/scores.json"
    nmfp_str = "—"
    if nmfp_p.exists():
        d = json.loads(nmfp_p.read_text())
        nmfp_str = f"{d['hr@1']:.4f}"
    hub_p = REPO / f"data/results/nafp/hubness_postproc/{cell}/scores.json"
    inv_str = csls_str = "—"
    if hub_p.exists():
        d = json.loads(hub_p.read_text())
        r = d.get("results", {})
        if "inv_softmax" in r: inv_str = f"{r['inv_softmax']['hr@1']:.4f}"
        if "csls" in r: csls_str = f"{r['csls']['hr@1']:.4f}"
    lines.append(f"| {cell} | {base} | {nmfp_str} | {inv_str} | {csls_str} |")

lines.append("")
lines.append("## Failure-mode breakdown (NAFP vs NMFP same-artist misses)")
lines.append("")
for L in [1, 3, 5, 10]:
    cell = f"main_{L}s"
    cmp_p = REPO / f"data/results/nafp/nmfp_eval/comparison/{cell}.json"
    if not cmp_p.exists():
        continue
    d = json.loads(cmp_p.read_text())
    lines.append(f"### {cell}")
    lines.append("")
    lines.append(f"- NAFP: HR@1={d['nafp']['hr@1']:.4f}  n_miss={d['nafp']['n_miss']}  same-artist miss={d['nafp']['n_miss_same_artist']}")
    lines.append(f"- NMFP: HR@1={d['nmfp']['hr@1']:.4f}  n_miss={d['nmfp']['n_miss']}  same-artist miss={d['nmfp']['n_miss_same_artist']}")
    mc = d['mcnemar']
    lines.append(f"- McNemar: b={mc['b']}  c={mc['c']}  Δhits={mc['delta']:+d}  p={mc['p']:.4f}")
    bonf = mc['p'] < 0.05/8
    lines.append(f"- Bonferroni-significant (α/8=0.00625): {'YES' if bonf else 'no'}")
    lines.append(f"- Queries fixed by NMFP: {len(d['fixed_by_nmfp'])}")
    lines.append(f"- Queries broken by NMFP: {len(d['broken_by_nmfp'])}")
    lines.append("")

lines.append("## What this means")
lines.append("")
lines.append("Decision tree based on results:")
lines.append("")
lines.append("- **If NMFP beats baseline AND mechanism is artist-hub:** add NMFP as system #5,")
lines.append("  cite Araz et al., write up as cross-domain benchmark contribution.")
lines.append("- **If NMFP does NOT beat baseline:** recipe doesn't transfer to non-Western tonal")
lines.append("  music. Publishable negative result. Focus remaining time on hubness post-proc.")
lines.append("- **If hubness (CSLS / InvSoftmax) beats baseline:** mechanism is geometry, not")
lines.append("  data. Publishable inference-time fix for non-Western music fingerprinting.")
lines.append("")
lines.append("Detailed comparison JSON: `data/results/nafp/nmfp_eval/comparison/`")
lines.append("All intermediate logs: `data/results/nafp/nmfp_eval/overnight.log`")

out.write_text("\n".join(lines))
print(f"WROTE {out}")
PY

log "=== OVERNIGHT PIPELINE COMPLETE ==="
log "See $SUMMARY for headline numbers"
log "Full log: $LOG"
