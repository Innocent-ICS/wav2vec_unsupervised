#!/bin/bash

# ============================================================
#  run_wav2vec_mac.sh — Full wav2vec-U data pipeline for macOS
#
#  This runs the complete preprocessing pipeline:
#    1. Regenerate manifests (with correct local paths)
#    2. Sample 10% of train data
#    3. Silence detection (rVADfast)
#    4. Silence removal
#    5. Audio feature extraction + clustering (prepare_audio)
#    6. Text phonemization + KenLM LM (prepare_text)
#
#  Usage:
#    bash run_wav2vec_mac.sh [--force-reset]
#
#  Options:
#    --force-reset   Wipe the progress checkpoint and restart all steps
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure Homebrew tools (wget, etc.) are on PATH
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

source "$SCRIPT_DIR/wav2vec_functions.sh"

# ---- Handle --force-reset flag ----
if [[ "${1}" == "--force-reset" ]]; then
    log "Flag --force-reset detected: clearing progress checkpoint..."
    rm -f "$CHECKPOINT_FILE"
    log "Checkpoint cleared. All steps will re-run."
fi

# ==================== PATH CONFIGURATION ====================
# LibriSpeech audio directories (local Mac paths)
TRAIN_DATASETS="$DATA_ROOT/LibriSpeech/train-clean-100"
VAL_DATASETS="$DATA_ROOT/LibriSpeech/dev-clean"
TEST_DATASETS="$DATA_ROOT/LibriSpeech/dev-clean"   # dev-clean serves as test too

# LibriSpeech LM text corpus (download if missing)
LM_TEXT_DIR="$DATA_ROOT/lm_data"
UNLABELLED_TEXT="$LM_TEXT_DIR/librispeech-lm-norm.txt"

# ==================== PRE-FLIGHT CHECKS ====================

log "============================================"
log " wav2vec-U  PIPELINE  (macOS M2  —  10% data)"
log "============================================"
log "Train audio  : $TRAIN_DATASETS"
log "Val audio    : $VAL_DATASETS"
log "Text corpus  : $UNLABELLED_TEXT"
log "Manifests    : $MANIFEST_DIR"
log "Non-sil audio: $NONSIL_AUDIO"
log "Clustering   : $CLUSTERING_DIR"
log "Text output  : $TEXT_OUTPUT"
log ""

# Check pretrained model
if [ ! -f "$MODEL" ]; then
    log "ERROR: Pre-trained model not found at $MODEL"
    log "Please download wav2vec_vox_new.pt to $SCRIPT_DIR/pre-trained/"
    exit 1
fi
log "Pre-trained model : $MODEL ✓"

# Check LM text corpus
if [ ! -f "$UNLABELLED_TEXT" ]; then
    log "LM text corpus not found at $UNLABELLED_TEXT"
    log "Attempting to download LibriSpeech LM corpus (~1.5GB compressed)..."
    mkdir -p "$LM_TEXT_DIR"
    GZ_FILE="$LM_TEXT_DIR/librispeech-lm-norm.txt.gz"
    curl -L --progress-bar -o "$GZ_FILE" \
        "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz" \
        || { log "ERROR: LM text download failed."; exit 1; }
    log "Decompressing..."
    gunzip "$GZ_FILE"
    log "LM text ready at $UNLABELLED_TEXT"
fi
log "LM text corpus    : $UNLABELLED_TEXT ✓"

# Activate venv
activate_venv
setup_path
create_dirs

log ""
log "==========================================="
log " STEP 1: Create full manifests"
log "==========================================="
create_manifests_train
create_manifests_val

# We don't have a separate test set; dev-clean is used for evaluation
log "Skipping separate test manifest — dev-clean is used for both val and test"

log ""
log "==========================================="
log " STEP 2: Sample 10% of training data"
log "==========================================="

SUBSAMPLE_STEP="subsample_10pct"
if is_completed "$SUBSAMPLE_STEP"; then
    log "Skipping subsampling (already completed)"
else
    FULL_TRAIN_TSV="$MANIFEST_DIR/train.tsv"
    FULL_LINE_COUNT=$(wc -l < "$FULL_TRAIN_TSV")
    TOTAL_UTTERANCES=$((FULL_LINE_COUNT - 1))   # subtract header line

    # 10% of utterances (rounded down), minimum 1
    SUBSAMPLE_N=$(( TOTAL_UTTERANCES / 10 ))
    [ "$SUBSAMPLE_N" -lt 1 ] && SUBSAMPLE_N=1

    log "Full train set: $TOTAL_UTTERANCES utterances"
    log "Sampling 10% : $SUBSAMPLE_N utterances (fixed seed for reproducibility)"

    # Keep header intact; shuffle the data lines and pick top SUBSAMPLE_N
    HEAD_LINE=$(head -1 "$FULL_TRAIN_TSV")
    BACKUP_FULL="$MANIFEST_DIR/train_full.tsv"

    # Back up the full manifest before overwriting
    cp "$FULL_TRAIN_TSV" "$BACKUP_FULL"
    log "Full manifest backed up to $BACKUP_FULL"

    # Subsample: header + reproducible random 10% of data rows
    {
        echo "$HEAD_LINE"
        tail -n +2 "$FULL_TRAIN_TSV" | \
            python3 -c "
import sys, random
random.seed(42)
lines = sys.stdin.readlines()
k = max(1, len(lines) // 10)
sampled = random.sample(lines, k)
sys.stdout.writelines(sampled)
"
    } > "$MANIFEST_DIR/train_10pct.tsv"

    # Use the subsampled TSV as the active train manifest
    cp "$MANIFEST_DIR/train_10pct.tsv" "$FULL_TRAIN_TSV"

    SAMPLED_COUNT=$(tail -n +2 "$FULL_TRAIN_TSV" | wc -l | tr -d ' ')
    log "Subsampled manifest written: $SAMPLED_COUNT utterances → $FULL_TRAIN_TSV"
    mark_completed "$SUBSAMPLE_STEP"
fi

log ""
log "==========================================="
log " STEP 3-4: Silence detection + removal"
log "==========================================="
create_rVADfast
remove_silence

log ""
log "==========================================="
log " STEP 5: Non-silence manifests"
log "==========================================="
create_manifests_nonsil_train
create_manifests_nonsil_val

log ""
log "==========================================="
log " STEP 6: Audio feature extraction (wav2vec)"
log "==========================================="
prepare_audio

log ""
log "==========================================="
log " STEP 7: Text preparation (G2P + KenLM)"
log "==========================================="
prepare_text

log ""
log "============================================"
log " PIPELINE COMPLETE"
log "============================================"
log ""
log "Next step: train the GAN"
log "  bash run_gans_mac.sh"
log ""
log "Audio features : $CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
log "Phone LM       : $TEXT_OUTPUT/phones/lm.phones.filtered.04.bin"
