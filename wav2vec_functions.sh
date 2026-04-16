#!/bin/bash

# Wav2Vec Unsupervised Pipeline Functions
# Adapted for macOS Apple Silicon M2

set -e
set -o pipefail

# ---- Source utils using absolute path (works when script is sourced from any dir) ----
_WAV2VEC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_WAV2VEC_SCRIPT_DIR/utils.sh"

# These are set by the caller (run_wav2vec_mac.sh) via utils.sh or overrides
# TRAIN_DATASETS, VAL_DATASETS, TEST_DATASETS, UNLABELLED_TEXT


# ==================== HELPER FUNCTIONS ====================

# Fix sflux() in rVADfast speechproc.py to return both s_flatness and n_frames.
# The root vads.py expects two return values; the stock rVADfast only returns one.
fixing_sflux() {
    local TARGET_FILE="$SPEECHPROCS"
    if [ -f "$TARGET_FILE" ]; then
        echo "Patching sflux() in $TARGET_FILE to return two values..."
        # macOS sed requires '' after -i
        sed -i '' '/def sflux/,/return/ s/^ *return .*/    return s_flatness, n_frames/' "$TARGET_FILE"
        echo "Patched sflux() return statement:"
        grep "return " "$TARGET_FILE" | head -5
        echo "Fix applied."
    else
        echo "WARNING: $TARGET_FILE not found — skipping sflux patch."
    fi
}

# Replace std::endl with "\n" in the Kaldi C++ file (pykaldi compatibility fix)
replace_std_endl() {
    local input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        echo "WARNING: '$input_file' not found — skipping std::endl patch."
        return 0
    fi
    # macOS sed requires '' after -i
    sed -i '' 's/std::endl/"\\n"/g' "$input_file"
    echo "Replaced std::endl in '$input_file'"
}

# Patch --sample-pct value in prepare_audio.sh (controls fraction used for k-means)
update_sample_pct() {
    sed -i '' -E "s/(--sample-pct[[:space:]]+)[0-9]*\.?[0-9]+/\1${NEW_SAMPLE_PCT}/g" "$PREPARE_AUDIO"
    echo "Updated '--sample-pct' to ${NEW_SAMPLE_PCT} in prepare_audio.sh"
}

# Patch --batch-size value in prepare_audio.sh
update_batch_size() {
    sed -i '' -E "s/(--batch-size[[:space:]]+)[0-9]+/\1${NEW_BATCH_SIZE}/g" "$PREPARE_AUDIO"
    echo "Updated '--batch-size' to ${NEW_BATCH_SIZE} in prepare_audio.sh"
}


# ==================== PIPELINE STEPS ====================

# Step 1a: Create train manifest (train.tsv) pointing to local audio files
create_manifests_train() {
    local step_name="create_manifests_train"

    if is_completed "$step_name"; then
        log "Skipping train manifest creation (already completed)"
        return 0
    fi

    log "Creating TRAIN data manifest from: $TRAIN_DATASETS"
    mark_in_progress "$step_name"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TRAIN_DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext flac \
        --valid-percent 0

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "TRAIN manifest created: $MANIFEST_DIR/train.tsv"
    else
        log "ERROR: TRAIN manifest creation failed"
        exit 1
    fi
}

# Step 1b: Create validation manifest (valid.tsv)
create_manifests_val() {
    local step_name="create_manifests_val"

    if is_completed "$step_name"; then
        log "Skipping validation manifest creation (already completed)"
        return 0
    fi

    log "Creating VALIDATION data manifest from: $VAL_DATASETS"
    mark_in_progress "$step_name"

    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_DIR/val_manifest.XXXXXX")

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$VAL_DATASETS" \
        --dest "$TEMP_VAL_DIR" \
        --ext flac \
        --valid-percent 1.0
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
        mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_DIR/valid.tsv"
        log "VALIDATION manifest created: $MANIFEST_DIR/valid.tsv"
        mark_completed "$step_name"
    else
        log "ERROR: VALIDATION manifest creation failed (exit code: $exit_code)"
        rm -rf "$TEMP_VAL_DIR"
        exit 1
    fi

    rm -rf "$TEMP_VAL_DIR"
}

# Step 1c: Create test manifest (test.tsv) — placed directly in MANIFEST_NONSIL_DIR
create_manifests_test() {
    local step_name="create_manifests_test"

    if is_completed "$step_name"; then
        log "Skipping test manifest creation (already completed)"
        return 0
    fi

    log "Creating TEST data manifest from: $TEST_DATASETS"
    mark_in_progress "$step_name"

    local MANIFEST_TEST_DIR="$DATA_ROOT/manifest_test_tmp"
    mkdir -p "$MANIFEST_TEST_DIR"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TEST_DATASETS" \
        --dest "$MANIFEST_TEST_DIR" \
        --ext flac \
        --valid-percent 0

    if [ $? -eq 0 ] && [ -f "$MANIFEST_TEST_DIR/train.tsv" ]; then
        mkdir -p "$MANIFEST_NONSIL_DIR"
        cp "$MANIFEST_TEST_DIR/train.tsv" "$MANIFEST_NONSIL_DIR/test.tsv"
        rm -rf "$MANIFEST_TEST_DIR"
        mark_completed "$step_name"
        log "TEST manifest created: $MANIFEST_NONSIL_DIR/test.tsv"
    else
        log "ERROR: TEST manifest creation failed"
        rm -rf "$MANIFEST_TEST_DIR"
        exit 1
    fi
}

# Step 2: Run rVADfast to detect silence boundaries → produces .vads files
create_rVADfast() {
    local step_name="create_rVADfast"

    # Apply the sflux() patch before checking completion
    fixing_sflux

    if is_completed "$step_name"; then
        log "Skipping rVADfast silence detection (already completed)"
        return 0
    fi

    log "Running rVADfast silence detection..."
    mark_in_progress "$step_name"

    python3 "$_WAV2VEC_SCRIPT_DIR/vads.py" -r "$RVAD_ROOT" \
        < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"

    python3 "$_WAV2VEC_SCRIPT_DIR/vads.py" -r "$RVAD_ROOT" \
        < "$MANIFEST_DIR/valid.tsv" > "$MANIFEST_DIR/valid.vads"

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "rVADfast silence detection complete"
    else
        log "ERROR: rVADfast failed"
        exit 1
    fi
}

# Step 3: Remove silence from audio files using the .vads boundaries
remove_silence() {
    local step_name="remove_silence"

    if is_completed "$step_name"; then
        log "Skipping silence removal (already completed)"
        return 0
    fi

    log "Removing silence from audio files..."
    mark_in_progress "$step_name"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" \
        --tsv "$MANIFEST_DIR/train.tsv" \
        --vads "$MANIFEST_DIR/train.vads" \
        --out "$NONSIL_AUDIO/train"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" \
        --tsv "$MANIFEST_DIR/valid.tsv" \
        --vads "$MANIFEST_DIR/valid.vads" \
        --out "$NONSIL_AUDIO/val"

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Silence removal complete"
    else
        log "ERROR: Silence removal failed"
        exit 1
    fi
}

# Step 4a: Create non-silence train manifest
create_manifests_nonsil_train() {
    local step_name="create_manifests_nonsil_train"

    if is_completed "$step_name"; then
        log "Skipping nonsil train manifest creation (already completed)"
        return 0
    fi

    log "Creating non-silence TRAIN manifest..."
    mark_in_progress "$step_name"

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/train" \
        --dest "$MANIFEST_NONSIL_DIR" \
        --ext flac \
        --valid-percent 0

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Non-silence TRAIN manifest created"
    else
        log "ERROR: Non-silence TRAIN manifest creation failed"
        exit 1
    fi
}

# Step 4b: Create non-silence validation manifest
create_manifests_nonsil_val() {
    local step_name="create_manifests_nonsil_val"

    if is_completed "$step_name"; then
        log "Skipping nonsil validation manifest creation (already completed)"
        return 0
    fi

    log "Creating non-silence VALIDATION manifest..."
    mark_in_progress "$step_name"

    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_NONSIL_DIR/val_manifest.XXXXXX")

    python3 "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/val" \
        --dest "$TEMP_VAL_DIR" \
        --ext flac \
        --valid-percent 1.0
    local exit_code=$?

    if [ $exit_code -eq 0 ] && [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
        mkdir -p "$MANIFEST_NONSIL_DIR"
        mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_NONSIL_DIR/valid.tsv"
        mark_completed "$step_name"
        log "Non-silence VALIDATION manifest created"
    else
        log "ERROR: Non-silence VALIDATION manifest creation failed (exit code: $exit_code)"
        rm -rf "$TEMP_VAL_DIR"
        exit 1
    fi

    rm -rf "$TEMP_VAL_DIR"
}

# Step 5: Extract wav2vec features, cluster (k-means), apply PCA → pseudo-phonemes
prepare_audio() {
    local step_name="prepare_audio"
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"

    # Patch sample-pct and batch-size in the upstream prepare_audio.sh
    update_sample_pct
    update_batch_size

    if is_completed "$step_name"; then
        log "Skipping audio preparation (already completed)"
        return 0
    fi

    log "Running audio feature extraction and clustering..."
    mark_in_progress "$step_name"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh" \
        "$MANIFEST_NONSIL_DIR" \
        "$CLUSTERING_DIR" \
        "$MODEL" \
        512 \
        14

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Audio preparation complete"
    else
        log "ERROR: Audio preparation failed"
        exit 1
    fi
}

# Step 6: Phonemize text corpus and build KenLM n-gram language model
prepare_text() {
    local step_name="prepare_text"
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"

    if is_completed "$step_name"; then
        log "Skipping text preparation (already completed)"
        return 0
    fi

    log "Preparing text data (phonemization + KenLM)..."
    mark_in_progress "$step_name"

    # Apply Kaldi C++ compatibility patch (std::endl → "\n")
    replace_std_endl "$ADD_SELF_LOOP_SIMPLE"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh" \
        "$LANG" \
        "$UNLABELLED_TEXT" \
        "$TEXT_OUTPUT" \
        "$MIN_PHONES" \
        "$PHONEMIZER" \
        "$FASTTEXT_LIB_MODEL" \
        0.25

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Text preparation complete"
    else
        log "ERROR: Text preparation failed"
        exit 1
    fi
}
