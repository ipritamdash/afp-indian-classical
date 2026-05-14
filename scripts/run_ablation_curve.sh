#!/bin/bash
# Re-run the ablation set across all 3 classical systems × {10s,5s,3s,1s} query
# lengths against the *new* 632-query manifest (post-F5 unicode fix).
# Uses warm DBs via --skip-index.
#
# Outputs to data/results/{system}/saraga_only_ablation_{1s,3s,5s}/  (10s overwrites
# the pre-existing saraga_only_ablation/).

set -euo pipefail
cd "$(dirname "$0")/.."

REFS_H="data/manifests/hindustani/refs.csv"
REFS_C="data/manifests/carnatic/refs.csv"

run_one () {
    local sys="$1" len="$2"
    local out
    local qh qc
    if [ "$len" = "10s" ]; then
        out="data/results/${sys}/saraga_only_ablation"
        qh="data/manifests/hindustani/queries_ablation.csv"
        qc="data/manifests/carnatic/queries_ablation.csv"
    else
        out="data/results/${sys}/saraga_only_ablation_${len}"
        qh="data/manifests/hindustani/queries_ablation_${len}.csv"
        qc="data/manifests/carnatic/queries_ablation_${len}.csv"
    fi
    local src_idx="data/results/${sys}/saraga_only_main/index_log.json"

    mkdir -p "$out"
    cp -f "$src_idx" "${out}/index_log.json"

    echo ""
    echo "=========================================================="
    echo "  $(date '+%H:%M:%S')  ${sys}  ×  ablation_${len}  →  ${out}"
    echo "=========================================================="
    uv run python "scripts/systems/${sys}_runner.py" \
        --refs "$REFS_H" "$REFS_C" \
        --queries "$qh" "$qc" \
        --output "$out" \
        --skip-index \
        --top-k 10

    # Score immediately
    uv run python scripts/score.py \
        --results "${out}/query_results.parquet" \
        --queries "$qh" "$qc" \
        --output "${out}/scores.json"

    echo "$(date '+%H:%M:%S')  DONE ${sys} × ablation_${len}"
}

START=$(date '+%s')
for SYS in olaf dejavu panako; do
    for LEN in 10s 5s 3s 1s; do
        run_one "$SYS" "$LEN" || { echo "STOPPING after failure"; exit 2; }
    done
done
END=$(date '+%s')
echo ""
echo "=========================================================="
echo "  ALL 12 ABLATION RUNS DONE in $((END - START)) sec ( $((($END - $START) / 60)) min )"
echo "=========================================================="
