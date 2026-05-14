#!/usr/bin/env bash
# Wait for seed42 to finish (its scores.json for the last cell ablation_10s appears),
# then run seed137 and seed2026 sequentially on Metal.
set -u
cd "/Users/prita/Desktop/Audio Fingerprinting/afp_bench"
LOG="data/results/nafp/recipe_v3_30ep/eval_remaining_seeds.log"
mkdir -p "$(dirname "$LOG")"
ts() { date -u +"%Y-%m-%d %H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== WAIT for seed42 to finish all 8 cells ==="
WAIT_FILE="data/results/nafp/recipe_v3_30ep/seed42_eval/ablation_10s/scores.json"
WAIT_START=$(date +%s)
while [ ! -f "$WAIT_FILE" ]; do
    sleep 60
    NOW=$(date +%s)
    MIN=$(( (NOW - WAIT_START) / 60 ))
    if [ "$MIN" -gt 120 ]; then
        log "TIMEOUT after 2h waiting for $WAIT_FILE"; exit 1
    fi
    if [ $(( MIN % 5 )) -eq 0 ] && [ "$MIN" -gt 0 ]; then
        log "  ...waiting ($MIN min elapsed)"
    fi
done
log "seed42 DONE. Final cell scores: $(cat $WAIT_FILE)"

for SEED in 137 2026; do
    log "=== seed${SEED} eval START ==="
    /tmp/venv_metal/bin/python scripts/nafp/recipe_v2_eval/eval.py \
        --ckpt-dir data/results/nafp/recipe_v3_30ep/seed${SEED} \
        --ckpt-name ckpt-30 \
        --cfg data/results/nafp/recipe_v3_30ep/recipe_v3.yaml \
        --out-dir data/results/nafp/recipe_v3_30ep/seed${SEED}_eval \
        --cells main_1s main_3s main_5s main_10s ablation_1s ablation_3s ablation_5s ablation_10s \
        >> "$LOG" 2>&1
    STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
        log "=== seed${SEED} DONE (exit 0) ==="
    else
        log "=== seed${SEED} FAILED (exit $STATUS) ==="
    fi
done

log "=== ALL 3 SEEDS COMPLETE ==="
# Summary
for SEED in 42 137 2026; do
    log "--- seed${SEED} HR@1 per cell ---"
    for CELL in main_1s main_3s main_5s main_10s ablation_1s ablation_3s ablation_5s ablation_10s; do
        F="data/results/nafp/recipe_v3_30ep/seed${SEED}_eval/${CELL}/scores.json"
        if [ -f "$F" ]; then
            HR=$(/tmp/venv_metal/bin/python -c "import json; d=json.load(open('$F')); print(f'hr@1={d[\"hr@1\"]:.4f} hits={d[\"n_hits\"]}/{d[\"n_queries\"]}')")
            log "  $CELL: $HR"
        else
            log "  $CELL: MISSING"
        fi
    done
done
