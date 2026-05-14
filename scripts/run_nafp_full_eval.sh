#!/bin/bash
# Full NAFP evaluation suite — after the main 10-s run has finished and the
# ref-embeddings index lives at data/results/nafp/saraga_only_main/.
#
# Reuses that index via --skip-index for every subsequent run:
#   * ablation 10s
#   * main 1s / 3s / 5s
#   * ablation 1s / 3s / 5s
#
# After each run, also calls score.py to write scores.json next to the parquet.
# Each --skip-index run only encodes the queries + searches FAISS, so it should
# complete in a few minutes (no re-indexing of the 691 k ref segments).

set -euo pipefail
cd "$(dirname "$0")/.."

INDEX_FROM="data/results/nafp/saraga_only_main"
CKPT_DIR="data/results/nafp/kaggle_output/logs/checkpoint"
CONFIG="scripts/nafp/upstream/config/default.yaml"
REFS_H="data/manifests/hindustani/refs.csv"
REFS_C="data/manifests/carnatic/refs.csv"

if [ ! -f "${INDEX_FROM}/ref_embs.mm" ] || [ ! -f "${INDEX_FROM}/ref_segment_lookup.parquet" ] || [ ! -f "${INDEX_FROM}/index_log.json" ]; then
    echo "FATAL: main inference not finished — missing index artefacts in ${INDEX_FROM}"
    exit 1
fi

run_one () {
    # $1 = variant: "main" or "ablation"
    # $2 = length: "10s", "5s", "3s", "1s"
    local variant="$1" len="$2"
    local out
    local qh qc

    if [ "$len" = "10s" ]; then
        suffix=""
    else
        suffix="_${len}"
    fi

    if [ "$variant" = "main" ]; then
        # Main 10s already done by the prior run — skip re-doing it
        if [ "$len" = "10s" ]; then
            echo ""
            echo "  $(date '+%H:%M:%S')  main × 10s already produced by the initial nafp_runner call — skipping"
            return 0
        fi
        out="data/results/nafp/saraga_only_main${suffix}"
        qh="data/manifests/hindustani/queries${suffix}.csv"
        qc="data/manifests/carnatic/queries${suffix}.csv"
    else
        out="data/results/nafp/saraga_only_ablation${suffix}"
        qh="data/manifests/hindustani/queries_ablation${suffix}.csv"
        qc="data/manifests/carnatic/queries_ablation${suffix}.csv"
    fi

    if [ ! -f "$qh" ] || [ ! -f "$qc" ]; then
        echo "FATAL: missing query manifest(s) for ${variant}×${len}: $qh  or  $qc"
        return 1
    fi
    mkdir -p "$out"

    echo ""
    echo "=========================================================="
    echo "  $(date '+%H:%M:%S')  NAFP × ${variant} × ${len}  →  ${out}"
    echo "=========================================================="
    uv run python scripts/systems/nafp_runner.py \
        --refs "$REFS_H" "$REFS_C" \
        --queries "$qh" "$qc" \
        --checkpoint-dir "$CKPT_DIR" \
        --checkpoint-name pipeline \
        --checkpoint-index 10 \
        --config "$CONFIG" \
        --output "$out" \
        --top-k 10 \
        --batch-size 256 \
        --skip-index \
        --index-from "$INDEX_FROM"

    uv run python scripts/score.py \
        --results "${out}/query_results.parquet" \
        --queries "$qh" "$qc" \
        --output "${out}/scores.json"

    echo "$(date '+%H:%M:%S')  DONE NAFP × ${variant} × ${len}"
}

# Also score the main 10s run that already finished
echo "=== score the already-done main 10s run ==="
uv run python scripts/score.py \
    --results "${INDEX_FROM}/query_results.parquet" \
    --queries "data/manifests/hindustani/queries.csv" "data/manifests/carnatic/queries.csv" \
    --output "${INDEX_FROM}/scores.json"

START=$(date '+%s')
for VARIANT in main ablation; do
    for LEN in 10s 5s 3s 1s; do
        run_one "$VARIANT" "$LEN" || { echo "STOPPING after failure"; exit 2; }
    done
done
END=$(date '+%s')
echo ""
echo "=========================================================="
echo "  ALL NAFP EVAL RUNS DONE in $((END - START)) sec ( $((($END - $START) / 60)) min )"
echo "=========================================================="
