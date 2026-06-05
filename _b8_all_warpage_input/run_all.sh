#!/usr/bin/env bash
# One-click runner: 4 train baselines x 2 inference datasets (main, secondary)
# = 8 inference jobs, each followed by a histogram comparison against the
# ground-truth dataset.
#
# Assumes the trained checkpoints exist at:
#   ./outputs/b8_all/warpage32_{1,2,3,4}.pth
# the inference *initial condition* datasets exist at:
#   ./dataset/infer_b8_main.h5
#   ./dataset/infer_b8_secondary.h5
# and the ground-truth *eval* datasets (used for histogram comparison) at:
#   ./dataset/eval_b8_main.h5
#   ./dataset/eval_b8_sec.h5
#
# Each inference's stdout is tee'd to ${LOG_ROOT}/infer_<tag>.log and the
# histogram step to ${LOG_ROOT}/hist_<tag>.log. The trained-model log files
# referenced by the configs (log_file_dir = b8_all/infer32_<i>_<ds>.log) are
# written by MeshGraphNets_main.py itself; these tee logs are just transcripts.
#
# Environment overrides:
#   PYTHON           = python interpreter (default: python)
#   LOG_ROOT         = directory for tee logs (default: outputs/b8_all/run_logs)
#   ONLY             = restrict to "infer" or "hist" only (default: both)
#   BASELINES        = space-separated baseline indices (default: "1 2 3 4")
#   DATASETS         = space-separated dataset tags (default: "main sec")
#
# Parallel run across GPUs (configs use gpu_ids 0,2,4,6 — non-overlapping):
#   BASELINES=1 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=2 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=3 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=4 bash _b8_all_warpage_input/run_all.sh &
#   wait

set -euo pipefail

PYTHON="${PYTHON:-python}"
ONLY="${ONLY:-both}"
BASELINES="${BASELINES:-1 2 3 4}"
DATASETS="${DATASETS:-main sec}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-outputs/b8_all/run_logs}"
mkdir -p "$LOG_ROOT"

eval_h5() {
    case "$1" in
        main) echo "dataset/eval_b8_main.h5" ;;
        sec)  echo "dataset/eval_b8_sec.h5" ;;
        *) echo "ERROR: unknown dataset tag '$1'" >&2; exit 2 ;;
    esac
}

run_one() {
    local idx=$1
    local ds=$2
    local tag="${idx}_${ds}"
    local cfg="_b8_all_warpage_input/config_infer${idx}_${ds}.txt"
    local rollout_dir="outputs/b8_all/infer32_${idx}_${ds}"
    local eval_ds
    eval_ds="$(eval_h5 "$ds")"

    if [ ! -f "$cfg" ]; then
        echo "[$tag] SKIP: config not found ($cfg)" >&2
        return 0
    fi

    if [ "$ONLY" != "hist" ]; then
        echo ""
        echo "=========================================="
        echo "[$tag] INFERENCE  cfg=$cfg"
        echo "=========================================="
        if ! "$PYTHON" MeshGraphNets_main.py --config "$cfg" 2>&1 \
                | tee "$LOG_ROOT/infer_${tag}.log"; then
            echo "[$tag] inference FAILED — see $LOG_ROOT/infer_${tag}.log" >&2
            return 1
        fi
    fi

    if [ "$ONLY" != "infer" ]; then
        echo ""
        echo "[$tag] HISTOGRAM  rollout_dir=$rollout_dir"
        if ! "$PYTHON" _b8_all_warpage_input/compare_histograms.py \
                --eval_dataset "$eval_ds" \
                --rollout_dir  "$rollout_dir" \
                --output       "$rollout_dir/histogram_compare.png" 2>&1 \
                | tee "$LOG_ROOT/hist_${tag}.log"; then
            echo "[$tag] histogram comparison FAILED — see $LOG_ROOT/hist_${tag}.log" >&2
            return 1
        fi
    fi
}

started=$(date +%s)
echo "One-click b8_all runner"
echo "  PYTHON     = $PYTHON"
echo "  ONLY       = $ONLY     (infer | hist | both)"
echo "  BASELINES  = $BASELINES"
echo "  DATASETS   = $DATASETS"
echo "  LOG_ROOT   = $LOG_ROOT"

for i in $BASELINES; do
    for ds in $DATASETS; do
        run_one "$i" "$ds"
    done
done

ended=$(date +%s)
echo ""
echo "All requested jobs complete in $((ended - started))s."
echo "Tee logs:        $LOG_ROOT/"
echo "Histogram PNGs:  outputs/b8_all/infer32_<i>_<ds>/histogram_compare.png"
