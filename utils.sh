#!/bin/bash

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs
# Adapted for macOS Apple Silicon M2 (arm64)

# Detect script directory robustly (works whether sourced or run directly)
if [[ -n "${BASH_SOURCE[0]}" ]]; then
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    _SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# Main directories
DIR_PATH="${_SCRIPT_DIR}"             # root of the project (local Mac path)
DATA_ROOT="$DIR_PATH/data"            # stores all generated pipeline data
FAIRSEQ_ROOT="$DIR_PATH/fairseq_"    # fairseq repository root
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"  # KenLM compiled binaries
VENV_PATH="$DIR_PATH/venv"           # Python virtual environment
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast"  # rVADfast source root

# Homebrew prefix (Apple Silicon default)
BREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

# Python command (macOS venv uses python3)
PYTHON="python3"

GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"


# ==================== HELPER FUNCTIONS ====================

# Fairseq file paths (with any project-specific patches)
SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py"


# Arguments/variables
NEW_SAMPLE_PCT=0.5
MIN_PHONES=3
NEW_BATCH_SIZE=32
PHONEMIZER="G2P"
LANG="en"

# Models
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin"
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt"

# Dataset specifics
DATASET_NAME="librispeech"

# Output directories (will be created if they don't exist)
MANIFEST_DIR="$DATA_ROOT/manifests"
NONSIL_AUDIO="$DATA_ROOT/processed_audio/"
MANIFEST_NONSIL_DIR="$DATA_ROOT/manifests_nonsil"
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET_NAME"
RESULTS_DIR="$DATA_ROOT/results/$DATASET_NAME"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME"
LOG_DIR="$DATA_ROOT/logs/$DATASET_NAME"
TEXT_OUTPUT="$DATA_ROOT/text"

# Checkpoint file to track pipeline progress
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"


# ==================== CHECKPOINT HELPERS ====================

log() {
    local message="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    mkdir -p "$LOG_DIR"
    echo "[$timestamp] $message" | tee -a "$LOG_DIR/pipeline.log"
}

is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

mark_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        # macOS-compatible sed: requires empty string after -i
        sed -i '' "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}


# ==================== ENV HELPERS ====================

setup_path() {
    export HYDRA_FULL_ERROR=1
    # On macOS, no LD_LIBRARY_PATH equivalent needed for KenLM/fairseq
    # DYLD_LIBRARY_PATH is used on macOS but fairseq finds libs via pip install
    export KENLM_ROOT="$KENLM_ROOT"
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export RVAD_ROOT="$RVAD_ROOT"
}

activate_venv() {
    if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
        # After activation, 'python3' should be available
        PYTHON="python3"
    else
        log "WARNING: Virtual environment not found at $VENV_PATH"
    fi
}

# Create all necessary directories
create_dirs() {
    mkdir -p "$MANIFEST_DIR" "$CLUSTERING_DIR" "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR" "$TEXT_OUTPUT" \
             "$GANS_OUTPUT_PHONES" "$NONSIL_AUDIO"
}
