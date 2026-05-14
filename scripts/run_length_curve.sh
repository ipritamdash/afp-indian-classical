#!/bin/bash
# Run all 3 classical systems against the 1s/3s/5s query variants using each
# system's existing warm DB (--skip-index). Sequential to stay within M5 RAM.
#
# Per-(system, length) artefacts go to:
#   data/results/{system}/saraga_only_main_{length}/
#     ├── index_log.json   (copied from saraga_only_main/, drives skip-index)
#     ├── query_results.parquet
#     ├── query_log.json
#     └── {system}_runner.log  (Olaf only — others log to stderr)
#
# Run:
#   bash scripts/run_length_curve.sh             (foreground, ~45 min)
#   bash scripts/run_length_curve.sh 2>&1 | tee data/results/length_curve.log

set -euo pipefail
cd "$(dirname "$0")/.."

REFS_H="data/manifests/hindustani/refs.csv"
REFS_C="data/manifests/carnatic/refs.csv"

run_one () {
    local sys="$1" len="$2"
    local out="data/results/${sys}/saraga_only_main_${len}"
    local src_idx="data/results/${sys}/saraga_only_main/index_log.json"
    local qh="data/manifests/hindustani/queries_${len}.csv"
    local qc="data/manifests/carnatic/queries_${len}.csv"

    if [ ! -f "$src_idx" ]; then
        echo "FATAL: ${sys} has no existing index_log.json at ${src_idx} — skipping"
        return 1
    fi
    if [ ! -f "$qh" ] || [ ! -f "$qc" ]; then
        echo "FATAL: missing query manifests for ${len} (${qh} or ${qc}) — run build_query_length_variants first"
        return 1
    fi

    mkdir -p "$out"
    cp -f "$src_idx" "${out}/index_log.json"

    echo ""
    echo "=========================================================="
    echo "  $(date '+%H:%M:%S')  ${sys}  ×  ${len}  →  ${out}"
    echo "=========================================================="
    uv run python "scripts/systems/${sys}_runner.py" \
        --refs "$REFS_H" "$REFS_C" \
        --queries "$qh" "$qc" \
        --output "$out" \
        --skip-index \
        --top-k 10
    echo "$(date '+%H:%M:%S')  DONE ${sys} × ${len}"
}

START=$(date '+%s')
for SYS in olaf dejavu panako; do
    for LEN in 1s 3s 5s; do
        run_one "$SYS" "$LEN" || { echo "STOPPING after failure"; exit 2; }
    done
done
END=$(date '+%s')
echo ""
echo "=========================================================="
echo "  ALL 9 RUNS DONE in $((END - START)) sec ( $((($END - $START) / 60)) min )"
echo "=========================================================="
