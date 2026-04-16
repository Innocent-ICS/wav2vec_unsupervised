#!/bin/bash

# ============================================================
#  eval_functions.sh — Evaluation functions for wav2vec-U
#  Adapted for macOS Apple Silicon M2
# ============================================================

set -e
set -o pipefail

# ---- Source utils using absolute path ----
_EVAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_EVAL_SCRIPT_DIR/utils.sh"

# The checkpoint path is set by run_eval.sh via MODEL_PATH=$DIR_PATH/$1
# This accepts either an absolute path or a path relative to DIR_PATH
if [ -n "$1" ]; then
    if [[ "$1" = /* ]]; then
        MODEL_PATH="$1"
    else
        MODEL_PATH="$DIR_PATH/$1"
    fi
else
    # run_eval.sh will validate this; set a default for sourcing context
    MODEL_PATH="${MODEL_PATH:-}"
fi


# ==================== EVALUATION FUNCTIONS ====================

# Viterbi decoding: produces phoneme-level transcriptions from the GAN model
# Output: phone transcription files saved to GANS_OUTPUT_PHONES
transcription_gans_viterbi() {
    export HYDRA_FULL_ERROR=1
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"
    export PYTHONPATH="$FAIRSEQ_ROOT:${PYTHONPATH:-}"

    log "Decoding with Viterbi..."
    log "  Audio features : $CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
    log "  Checkpoint     : $MODEL_PATH"
    log "  Text data      : $TEXT_OUTPUT/phones/"
    log "  Output dir     : $GANS_OUTPUT_PHONES"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" \
        --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
        --config-name viterbi \
        fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
        fairseq.task.data="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled" \
        fairseq.common_eval.path="$MODEL_PATH" \
        fairseq.task.text_data="$TEXT_OUTPUT/phones/" \
        fairseq.dataset.gen_subset=valid \
        fairseq.dataset.batch_size=1 \
        fairseq.dataset.num_workers=0 \
        fairseq.dataset.required_batch_size_multiple=1 \
        results_path="$GANS_OUTPUT_PHONES"

    log "Viterbi decoding complete. Results at: $GANS_OUTPUT_PHONES"
}
