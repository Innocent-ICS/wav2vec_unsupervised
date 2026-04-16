#!/bin/bash

# ============================================================
#  run_gans_mac.sh — GAN training for macOS Apple Silicon M2
#
#  Trains the wav2vec-U GAN model on top of precomputed audio
#  features and a phoneme-level language model.
#
#  Prerequisite: run_wav2vec_mac.sh must have completed first.
#
#  Usage:
#    bash run_gans_mac.sh [--force-reset]
#
#  Options:
#    --force-reset   Re-run GAN training even if already completed
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gans_functions.sh"

# ---- Handle --force-reset flag ----
if [[ "${1}" == "--force-reset" ]]; then
    log "Flag --force-reset detected: clearing GAN training checkpoint..."
    if [ -f "$CHECKPOINT_FILE" ]; then
        sed -i '' '/^train_gans/d' "$CHECKPOINT_FILE"
    fi
    log "GAN training checkpoint cleared."
fi

# ==================== PRE-FLIGHT CHECKS ====================

log "============================================"
log " wav2vec-U  GAN TRAINING  (macOS M2)"
log "============================================"

# Verify precomputed audio features exist
AUDIO_FEATURES="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
if [ ! -d "$AUDIO_FEATURES" ]; then
    log "ERROR: Audio features not found at $AUDIO_FEATURES"
    log "Please run 'bash run_wav2vec_mac.sh' first to generate features."
    exit 1
fi
log "Audio features : $AUDIO_FEATURES ✓"

# Verify phone LM exists
PHONE_LM="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin"
if [ ! -f "$PHONE_LM" ]; then
    log "ERROR: Phone language model not found at $PHONE_LM"
    log "Please run 'bash run_wav2vec_mac.sh' first to prepare text."
    exit 1
fi
log "Phone LM       : $PHONE_LM ✓"

# Verify phone data directory
PHONE_DATA="$TEXT_OUTPUT/phones/"
if [ ! -d "$PHONE_DATA" ]; then
    log "ERROR: Phone data directory not found at $PHONE_DATA"
    exit 1
fi
log "Phone data     : $PHONE_DATA ✓"

log ""
log "Mac-specific training settings:"
log "  Seeds        : 1 (range 0,1) — reduced from default 5 for speed"
log "  Max updates  : 30,000 — reduced from 150,000 for M2 feasibility"
log "  Batch size   : 8 — fits within 24GB unified memory"
log "  Num workers  : 0 — avoids macOS multiprocessing fork issues"
log "  World size   : 1 — single process, no NCCL"
log ""

# Activate venv and set paths
activate_venv
setup_path
create_dirs

log "==========================================="
log " Starting GAN training..."
log "==========================================="

train_gans

log ""
log "============================================"
log " GAN TRAINING COMPLETE"
log "============================================"
log ""
log "Checkpoints saved to : $RESULTS_DIR"
log ""
log "To find the best checkpoint (lowest unsupervised metric):"
log "  ls -lt $RESULTS_DIR/checkpoint*.pt | head -5"
log ""
log "To generate phone transcriptions with the best checkpoint:"
log "  bash run_eval.sh <checkpoint_name.pt>"
