#!/bin/bash

# ============================================================
#  run_eval.sh — Evaluate wav2vec-U GAN with Viterbi decoding
#
#  Usage:
#    bash run_eval.sh <path/to/checkpoint.pt>
#
#  Example:
#    bash run_eval.sh data/results/librispeech/checkpoint_best.pt
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/eval_functions.sh"

create_dirs
activate_venv
setup_path

log "============================================"
log " wav2vec-U  EVALUATION  (macOS M2)"
log "============================================"
log "Checkpoint : $MODEL_PATH"

# Validate checkpoint path was given and exists
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    log "ERROR: Checkpoint file not found: ${MODEL_PATH:-<not provided>}"
    log "Usage: bash run_eval.sh <path/to/checkpoint.pt>"
    exit 1
fi

# The GAN checkpoints live in a multirun directory structure:
#   data/results/librispeech/
#       <date>/<time>/0/   ← seed 0
#           checkpoint_best.pt
#           checkpoint_last.pt
#
# Pass the exact .pt file path as the argument.

log "Running Viterbi decoding on test set..."
transcription_gans_viterbi

log ""
log "============================================"
log " EVALUATION COMPLETE"
log "============================================"
log "Phone transcriptions saved to: $GANS_OUTPUT_PHONES"
