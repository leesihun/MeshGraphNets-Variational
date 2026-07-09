#!/usr/bin/env bash
# One-click TRAINING runner for the FM 4-factor sweep (cells 1..8).
#
# Each config_train{n}.txt pins a distinct GPU (gpu_ids 0..7), so by default all
# requested cells train in PARALLEL, one per GPU. Each cell's stdout/stderr is
# redirected to ${LOG_ROOT}/train_{n}.log (not tee'd — 8 concurrent streams would
# interleave; use `tail -f` to watch a cell).
#
# Cold multiscale cache is safe under parallel start: an exclusive O_EXCL lock
# (general_modules/multiscale_cache.py) lets exactly one process build the
# hierarchy while the rest wait, then it is atomically renamed into place.
#
# The trained checkpoints land at ./outputs/b8_all/warpage_train{n}_<tag>.pth,
# ready for _b8_all_warpage_input/run_all.sh (inference + histogram compare).
#
# Environment overrides:
#   PYTHON     = python interpreter (default: python)
#   LOG_ROOT   = directory for transcript logs (default: outputs/b8_all/run_logs)
#   TRAINS     = space-separated cell indices (default: "1 2 3 4 5 6 7 8")
#   PARALLEL   = 1 launch all at once then wait (default); 0 run sequentially
#                (one at a time — use when GPUs are shared or memory is tight)
#
# Usage:
#   bash _b8_all_warpage_input/run_train_all.sh
#   TRAINS="3 4 7 8" bash _b8_all_warpage_input/run_train_all.sh   # subset
#   PARALLEL=0       bash _b8_all_warpage_input/run_train_all.sh   # sequential
#   watch progress:  tail -f outputs/b8_all/run_logs/train_3.log

# NOT `set -e`: we collect per-cell failures and keep the batch going.
set -uo pipefail

PYTHON="${PYTHON:-python}"
TRAINS="${TRAINS:-1 2 3 4 5 6 7 8}"
PARALLEL="${PARALLEL:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-outputs/b8_all/run_logs}"
mkdir -p "$LOG_ROOT"

train_one() {
    local idx=$1
    local cfg="_b8_all_warpage_input/config_train${idx}.txt"
    local log="$LOG_ROOT/train_${idx}.log"
    if [ ! -f "$cfg" ]; then
        echo "[train$idx] SKIP: config not found ($cfg)" >&2
        return 0
    fi
    echo "[train$idx] START  cfg=$cfg  -> $log"
    if "$PYTHON" MeshGraphNets_main.py --config "$cfg" > "$log" 2>&1; then
        echo "[train$idx] DONE"
        return 0
    else
        echo "[train$idx] FAILED (exit $?) — see $log" >&2
        return 1
    fi
}

started=$(date +%s)
echo "FM 4-factor sweep — training runner"
echo "  PYTHON   = $PYTHON"
echo "  TRAINS   = $TRAINS"
echo "  PARALLEL = $PARALLEL"
echo "  LOG_ROOT = $LOG_ROOT"

rc=0
if [ "$PARALLEL" = "1" ]; then
    pids=()
    idxs=()
    for i in $TRAINS; do
        train_one "$i" &
        pids+=("$!")
        idxs+=("$i")
        echo "  launched train$i (pid $!)"
    done
    for k in "${!pids[@]}"; do
        if ! wait "${pids[$k]}"; then
            echo "train${idxs[$k]} exited non-zero" >&2
            rc=1
        fi
    done
else
    for i in $TRAINS; do
        train_one "$i" || rc=1
    done
fi

ended=$(date +%s)
echo ""
echo "All requested trainings finished in $((ended - started))s (rc=$rc)."
echo "Transcripts:  $LOG_ROOT/train_<n>.log"
echo "Checkpoints:  outputs/b8_all/warpage_train<n>_<tag>.pth"
echo "Next:         bash _b8_all_warpage_input/run_all.sh   (inference + histograms)"
exit $rc
