#!/bin/bash

# GAN Training Functions — Wav2Vec Unsupervised
# Adapted for macOS Apple Silicon M2

set -e
set -o pipefail

# ---- Source utils using absolute path ----
_GANS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_GANS_SCRIPT_DIR/utils.sh"


# ==================== GAN TRAINING ====================
#
# Trains the wav2vec-U GAN on top of precomputed audio features.
#
# Mac-specific adjustments:
#   - common.seed=range(0,1)         → single seed only (5 seeds too slow on CPU/MPS)
#   - dataset.num_workers=0          → avoids multiprocessing issues on macOS
#   - distributed_training.distributed_world_size=1  → single-process (no NCCL)
#   - max_update=30000               → reduced from default 150000 for feasibility
#   - dataset.batch_size=8           → smaller batch fits in 24GB unified memory
#   - +common.cpu=true               → forces CPU mode if MPS ops are unsupported
#
train_gans() {
    local step_name="train_gans"

    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"
    # Fix: original had a leading '/' typo before $DIR_PATH
    export PYTHONPATH="$FAIRSEQ_ROOT:${PYTHONPATH:-}"

    if is_completed "$step_name"; then
        log "Skipping GAN training (already completed)"
        return 0
    fi

    log "Starting GAN training..."
    log "  Audio features : $CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
    log "  Text data      : $TEXT_OUTPUT/phones/"
    log "  KenLM model    : $TEXT_OUTPUT/phones/lm.phones.filtered.04.bin"
    log "  Checkpoint dir : $RESULTS_DIR"
    log "  Log file       : $RESULTS_DIR/training.log"
    mark_in_progress "$step_name"

    mkdir -p "$RESULTS_DIR"

    # Detect MPS availability and set device accordingly
    local USE_CPU_FLAG=""
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        log "MPS (Metal) backend available — using GPU acceleration"
        # MPS is auto-detected by PyTorch; no explicit flag needed for fairseq
        # but some fairseq ops may fall back to CPU silently
    else
        log "MPS not available — running on CPU"
        USE_CPU_FLAG="+common.cpu=true"
    fi

    PYTHONPATH="$FAIRSEQ_ROOT" \
    PREFIX=w2v_unsup_gan_xp \
    "$VENV_PATH/bin/fairseq-hydra-train" \
        -m \
        --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
        --config-name w2vu \
        task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
        task.text_data="$TEXT_OUTPUT/phones/" \
        task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
        common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
        checkpoint.save_dir="$RESULTS_DIR" \
        model.code_penalty=6,10 \
        model.gradient_penalty=0.5,1.0 \
        model.smoothness_weight=1.5 \
        'common.seed=range(0,1)' \
        +optimizer.groups.generator.optimizer.lr="[0.00004]" \
        +optimizer.groups.discriminator.optimizer.lr="[0.00002]" \
        ~optimizer.groups.generator.optimizer.amsgrad \
        ~optimizer.groups.discriminator.optimizer.amsgrad \
        optimization.max_update=30000 \
        dataset.batch_size=8 \
        dataset.num_workers=0 \
        distributed_training.distributed_world_size=1 \
        common.log_interval=100 \
        $USE_CPU_FLAG \
        2>&1 | tee "$RESULTS_DIR/training.log"

    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        mark_completed "$step_name"
        log "GAN training complete. Checkpoints saved to $RESULTS_DIR"
    else
        log "ERROR: GAN training failed (exit code: $exit_code)"
        exit 1
    fi
}
