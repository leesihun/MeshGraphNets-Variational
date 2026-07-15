#!/usr/bin/env bash
# One-click runner: 8 train baselines x 2 inference datasets (main, secondary)
# = 16 inference jobs, each followed by a histogram comparison against the
# ground-truth dataset.
#
# Assumes the trained checkpoints exist at:
#   ./outputs/b8_all/warpage_train{1..8}_*.pth
# the inference *initial condition* datasets exist at:
#   ./dataset/infer_b8_main.h5
#   ./dataset/infer_b8_secondary.h5
# and the ground-truth *eval* datasets (used for histogram comparison) at:
#   ./dataset/eval_b8_main.h5
#   ./dataset/eval_b8_sec.h5
#
# Each inference's stdout is tee'd to ${LOG_ROOT}/infer_<tag>.log and the
# histogram step to ${LOG_ROOT}/hist_<tag>.log. The trained-model log files
# referenced by the configs (log_file_dir = b8_all/infer_train<i>_<ds>.log) are
# written by MeshGraphNets_main.py itself; these tee logs are just transcripts.
# After all jobs complete, mu/sigma stats from each histogram run are compiled
# into ${LOG_ROOT}/summary_stats.txt.
#
# Environment overrides:
#   PYTHON           = python interpreter (default: python)
#   LOG_ROOT         = directory for tee logs (default: outputs/b8_all/run_logs)
#   ONLY             = "infer" | "hist" | "diag" | "both" (default: both = infer+hist)
#   DIAG             = 1 to also run diag_prior_spread.py after infer/hist (default: 0)
#   TRIM             = quantile to drop from each tail in the histogram compare
#                      (e.g. 0.02 -> central 96%; ignores artifact/spurious tails; default 0)
#   BASELINES        = space-separated baseline indices (default: "1 2 3 4 5 6 7 8")
#   DATASETS         = space-separated dataset tags (default: "main sec")
#   VAE_BATCH_SIZE   = integer or "auto" (default: auto)
#   VAE_BATCH_VRAM_FRACTION = free-VRAM fraction for auto batching (default: 0.70)
#   VAE_BATCH_SIZE_MIN = lower bound after auto probe (default: 1)
#   VAE_BATCH_SIZE_MAX = upper bound after auto probe; 0 = uncapped (default: 0)
#   NUM_VAE_SAMPLES  = optional override for quick runs
#   PLOT_JOBS        = workers for mesh PNG plotting (default: nproc/2)
#   PLOT_DPI         = mesh PNG DPI (default: 80)
#   HIST_JOBS        = workers for histogram rollout loading (default: 8)
#
# diag_prior_spread.py reports, per checkpoint+type, the warpage-amplitude spread
# under posterior / prior / full-cov z-sources + the mu_q eigen-spectrum — i.e.
# whether under-dispersion is a prior problem (fixable) or a decoder ceiling.
#
# Parallel run across GPUs (configs use gpu_ids 0-7 — non-overlapping):
#   BASELINES=1 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=2 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=3 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=4 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=5 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=6 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=7 bash _b8_all_warpage_input/run_all.sh &
#   BASELINES=8 bash _b8_all_warpage_input/run_all.sh &
#   wait

set -euo pipefail

PYTHON="${PYTHON:-python}"
ONLY="${ONLY:-both}"
DIAG="${DIAG:-0}"
TRIM="${TRIM:-0}"   # e.g. 0.02 -> drop bottom/top 2% of each distribution in the histogram compare
BASELINES="${BASELINES:-1 2 3 4 5 6 7 8}" 
DATASETS="${DATASETS:-main sec}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-auto}"
VAE_BATCH_VRAM_FRACTION="${VAE_BATCH_VRAM_FRACTION:-0.70}"
VAE_BATCH_SIZE_MIN="${VAE_BATCH_SIZE_MIN:-1}"
VAE_BATCH_SIZE_MAX="${VAE_BATCH_SIZE_MAX:-0}"
NUM_VAE_SAMPLES="${NUM_VAE_SAMPLES:-}"
NPROC="$(nproc 2>/dev/null || echo 8)"
PLOT_JOBS="${PLOT_JOBS:-$(( NPROC / 2 > 0 ? NPROC / 2 : 1 ))}"
PLOT_DPI="${PLOT_DPI:-80}"
HIST_JOBS="${HIST_JOBS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_ROOT="${LOG_ROOT:-outputs/b8_all/run_logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
RUNTIME_CONFIG_ROOT="$LOG_ROOT/runtime_configs/$RUN_ID"
mkdir -p "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"

eval_h5() {
    case "$1" in
        main) echo "dataset/eval_b8_main.h5" ;;
        sec)  echo "dataset/eval_b8_sec.h5" ;;
        *) echo "ERROR: unknown dataset tag '$1'" >&2; exit 2 ;;
    esac
}

# Ground-truth amplitude (max-min z_disp) mu/sigma per type, for the diag verdict
# line. Measured from the eval datasets; override here if the datasets change.
gt_stats() {
    case "$1" in
        main) echo "701 280" ;;
        sec)  echo "749 360" ;;
        *)    echo "0 1" ;;
    esac
}

runtime_config() {
    local source_cfg=$1
    local tag=$2
    local out_cfg="$RUNTIME_CONFIG_ROOT/config_infer${tag}.txt"

    cp "$source_cfg" "$out_cfg"
    {
        echo ""
        echo "% Runtime overrides from run_all.sh"
        echo "% Source config: $source_cfg"
        echo "% Run ID: $RUN_ID"
        if [ -n "$NUM_VAE_SAMPLES" ]; then
            echo "num_vae_samples $NUM_VAE_SAMPLES"
        fi
        echo "vae_batch_size $VAE_BATCH_SIZE"
        echo "vae_batch_vram_fraction $VAE_BATCH_VRAM_FRACTION"
        echo "vae_batch_size_min $VAE_BATCH_SIZE_MIN"
        echo "vae_batch_size_max $VAE_BATCH_SIZE_MAX"
    } >> "$out_cfg"

    echo "$out_cfg"
}

diag_one() {
    local idx=$1
    local ds=$2
    local tag="${idx}_${ds}"
    local cfg="_b8_all_warpage_input/config_infer${idx}_${ds}.txt"
    local eval_ds gt
    eval_ds="$(eval_h5 "$ds")"
    gt="$(gt_stats "$ds")"

    if [ ! -f "$cfg" ]; then
        echo "[$tag] SKIP diag: config not found ($cfg)" >&2
        return 0
    fi
    echo ""
    echo "[$tag] DIAGNOSTIC  (posterior / prior / fullcov + mu_q eigen-spectrum)"
    "$PYTHON" diag_prior_spread.py \
        --config "$cfg" --gt_dataset "$eval_ds" \
        --gt_mu "${gt% *}" --gt_sigma "${gt#* }" 2>&1 \
        | tee "$LOG_ROOT/diag_${tag}.log" \
        || echo "[$tag] diagnostic FAILED — see $LOG_ROOT/diag_${tag}.log" >&2
}

run_one() {
    local idx=$1
    local ds=$2
    local tag="${idx}_${ds}"
    local source_cfg="_b8_all_warpage_input/config_infer${idx}_${ds}.txt"
    local cfg="$source_cfg"
    local rollout_dir="outputs/b8_all/infer_train${idx}_${ds}"
    local eval_ds
    eval_ds="$(eval_h5 "$ds")"

    if [ ! -f "$source_cfg" ]; then
        echo "[$tag] SKIP: config not found ($source_cfg)" >&2
        return 0
    fi

    if [ "$ONLY" = "both" ] || [ "$ONLY" = "infer" ]; then
        cfg="$(runtime_config "$source_cfg" "$tag")"
        echo ""
        echo "=========================================="
        echo "[$tag] INFERENCE  cfg=$cfg  source=$source_cfg"
        echo "=========================================="
        if ! "$PYTHON" MeshGraphNets_main.py --config "$cfg" 2>&1 \
                | tee "$LOG_ROOT/infer_${tag}.log"; then
            echo "[$tag] inference FAILED — see $LOG_ROOT/infer_${tag}.log" >&2
            return 1
        fi

        echo ""
        echo "[$tag] MESH PLOTS  rollout_dir=$rollout_dir"
        if ! "$PYTHON" _b8_all_warpage_input/plot_rollout_meshes.py \
                --rollout_dir "$rollout_dir" \
                --jobs "$PLOT_JOBS" \
                --dpi "$PLOT_DPI" 2>&1 \
                | tee "$LOG_ROOT/plot_${tag}.log"; then
            echo "[$tag] mesh plotting FAILED (non-fatal) — see $LOG_ROOT/plot_${tag}.log" >&2
        fi
    fi

    if [ "$ONLY" = "both" ] || [ "$ONLY" = "hist" ]; then
        echo ""
        echo "[$tag] HISTOGRAM  rollout_dir=$rollout_dir"
        if ! "$PYTHON" _b8_all_warpage_input/compare_histograms.py \
                --eval_dataset "$eval_ds" \
                --rollout_dir  "$rollout_dir" \
                --trim_quantile "$TRIM" \
                --jobs "$HIST_JOBS" \
                --output       "$rollout_dir/histogram_compare.png" 2>&1 \
                | tee "$LOG_ROOT/hist_${tag}.log"; then
            echo "[$tag] histogram comparison FAILED — see $LOG_ROOT/hist_${tag}.log" >&2
            return 1
        fi
    fi

    if [ "$ONLY" = "diag" ] || [ "$DIAG" = "1" ]; then
        diag_one "$idx" "$ds"
    fi
}

started=$(date +%s)
echo "One-click b8_all runner"
echo "  PYTHON     = $PYTHON"
echo "  ONLY       = $ONLY     (infer | hist | both)"
echo "  BASELINES  = $BASELINES"
echo "  DATASETS   = $DATASETS"
echo "  LOG_ROOT   = $LOG_ROOT"
echo "  RUN_ID     = $RUN_ID"
echo "  VAE_BATCH_SIZE = $VAE_BATCH_SIZE"
echo "  VAE_BATCH_VRAM_FRACTION = $VAE_BATCH_VRAM_FRACTION"
echo "  VAE_BATCH_SIZE_MIN/MAX = $VAE_BATCH_SIZE_MIN / $VAE_BATCH_SIZE_MAX"
if [ -n "$NUM_VAE_SAMPLES" ]; then
    echo "  NUM_VAE_SAMPLES = $NUM_VAE_SAMPLES"
fi
echo "  PLOT_JOBS  = $PLOT_JOBS"
echo "  HIST_JOBS  = $HIST_JOBS"

for i in $BASELINES; do
    for ds in $DATASETS; do
        run_one "$i" "$ds"
    done
done

ended=$(date +%s)
echo ""
echo "All requested jobs complete in $((ended - started))s."
echo "Tee logs:        $LOG_ROOT/"
echo "Runtime configs: $RUNTIME_CONFIG_ROOT/"
echo "Histogram PNGs:  outputs/b8_all/infer_train<i>_<ds>/histogram_compare.png"

# ── mu / sigma summary ──────────────────────────────────────────────────────
if [ "$ONLY" = "both" ] || [ "$ONLY" = "hist" ]; then
    SUMMARY="$LOG_ROOT/summary_stats.txt"
    {
        printf "# Spread summary (mu/sigma + tail/outlier metrics) — %s\n" "$(date)"
        hdr="%-14s  %11s %11s  %11s %11s %11s  %8s %8s %11s\n"
        printf "$hdr" "tag" "GT_mu" "GT_sig" "Gen_mu" "Gen_sig" "Gen_max" \
               "p99_cov" "g>GTp99" "wass"
        printf "$hdr" "------" "-----" "-----" "-----" "-----" "-----" \
               "-----" "-----" "-----"
        for i in $BASELINES; do
            for ds in $DATASETS; do
                tag="${i}_${ds}"
                log="$LOG_ROOT/hist_${tag}.log"
                [ -f "$log" ] || continue
                g="grep -o"
                gt_mu=$(  grep "^HIST_STATS  GT " "$log"  | $g 'mu=[^ ]*'    | cut -d= -f2 )
                gt_sig=$( grep "^HIST_STATS  GT " "$log"  | $g 'sigma=[^ ]*' | cut -d= -f2 )
                gn_mu=$(  grep "^HIST_STATS  Gen" "$log"  | $g 'mu=[^ ]*'    | cut -d= -f2 )
                gn_sig=$( grep "^HIST_STATS  Gen" "$log"  | $g 'sigma=[^ ]*' | cut -d= -f2 )
                gn_max=$( grep "^HIST_STATS  Gen" "$log"  | $g 'max=[^ ]*'   | cut -d= -f2 )
                p99c=$(   grep "^HIST_COMPARE"    "$log"  | $g 'p99_cov=[^ ]*'         | cut -d= -f2 )
                gp99=$(   grep "^HIST_COMPARE"    "$log"  | $g 'gen_frac_gt_p99=[^ ]*' | cut -d= -f2 )
                wass=$(   grep "^HIST_COMPARE"    "$log"  | $g 'wasserstein=[^ ]*'     | cut -d= -f2 )
                printf "$hdr" "$tag" "$gt_mu" "$gt_sig" "$gn_mu" "$gn_sig" "$gn_max" \
                       "$p99c" "$gp99" "$wass"
            done
        done
    } > "$SUMMARY"
    echo "Summary stats:   $SUMMARY"
fi
