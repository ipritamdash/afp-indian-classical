#!/usr/bin/env bash
# Run NMFP-ckpt-100 on all 4 ablation cells. Reuses NMFP index built last night.
set -u
cd "/Users/prita/Desktop/Audio Fingerprinting/afp_bench"
LOG="data/results/nafp/nmfp_eval/ablation_run.log"
ts() { date -u +"%Y-%m-%d %H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== NMFP ABLATION CELLS START ==="

# 4 cells with their specific manifest filenames
declare -a CELLS=(
  "ablation_1s:queries_ablation_1s.csv:queries_ablation_1s.csv"
  "ablation_3s:queries_ablation_3s.csv:queries_ablation_3s.csv"
  "ablation_5s:queries_ablation_5s.csv:queries_ablation_5s.csv"
  "ablation_10s:queries_ablation.csv:queries_ablation.csv"
)

for entry in "${CELLS[@]}"; do
    IFS=':' read -r CELL HQ CQ <<< "$entry"
    log "Cell $CELL: H=$HQ  C=$CQ"
    uv run python scripts/nafp/nmfp_eval/run_nmfp_native.py \
        --cell "$CELL" \
        --queries "data/manifests/hindustani/$HQ" "data/manifests/carnatic/$CQ" \
        --skip-index >> "$LOG" 2>&1
    SCORE_FILE="data/results/nafp/nmfp_eval/saraga_${CELL}/scores.json"
    if [ -f "$SCORE_FILE" ]; then
        HR=$(uv run python -c "import json; d=json.load(open('$SCORE_FILE')); print(f\"hr@1={d['hr@1']:.4f}  hits={d['n_hits']}/{d['n_queries']}\")" 2>/dev/null)
        log "  OK: $CELL  $HR"
    else
        log "  FAIL: no scores.json for $CELL"
    fi
done

log "=== NMFP ABLATION COMPLETE ==="
log "Per-cell scores.json files at data/results/nafp/nmfp_eval/saraga_ablation_{1,3,5,10}s/"
