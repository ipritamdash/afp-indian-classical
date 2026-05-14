#!/usr/bin/env bash
# Wait for current main_1s eval to finish (scores.json appears), then
# run all 7 remaining cells. They will reuse the already-built ref_embs.mm
# (eval.py has skip-if-exists logic).
set -u
cd "/Users/prita/Desktop/Audio Fingerprinting/afp_bench"
LOG="data/results/nafp/recipe_v2_10ep/followup.log"
ts() { date -u +"%Y-%m-%d %H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== AUTO-FOLLOWUP START — waiting for main_1s scores.json ==="
WAIT_FILE="data/results/nafp/recipe_v2_10ep/main_1s/scores.json"
START=$(date +%s)
while [ ! -f "$WAIT_FILE" ]; do
    sleep 60
    NOW=$(date +%s)
    MIN=$(( (NOW - START) / 60 ))
    if [ "$MIN" -gt 180 ]; then
        log "TIMEOUT after 3h waiting"; exit 1
    fi
    if [ $(( MIN % 10 )) -eq 0 ] && [ "$MIN" -gt 0 ]; then
        log "  ...waiting ($MIN min elapsed)"
    fi
done
log "main_1s DONE — $(cat "$WAIT_FILE")"

log "Launching 7 remaining cells (skip-index reuse)"
uv run python scripts/nafp/recipe_v2_eval/eval.py \
    --cells main_3s main_5s main_10s ablation_1s ablation_3s ablation_5s ablation_10s \
    >> "$LOG" 2>&1

log "=== AUTO-FOLLOWUP COMPLETE ==="
for c in main_3s main_5s main_10s ablation_1s ablation_3s ablation_5s ablation_10s; do
    f="data/results/nafp/recipe_v2_10ep/$c/scores.json"
    if [ -f "$f" ]; then
        log "$c: $(uv run python -c "import json; d=json.load(open('$f')); print(f'hr@1={d[\"hr@1\"]:.4f} hits={d[\"n_hits\"]}/{d[\"n_queries\"]}')" 2>/dev/null)"
    else
        log "$c: FAIL no scores.json"
    fi
done
