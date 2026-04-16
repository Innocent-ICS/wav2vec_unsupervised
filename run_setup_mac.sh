#!/bin/bash

# ============================================================
#  run_setup_mac.sh — One-time environment setup for macOS M2
#  Usage: bash run_setup_mac.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_functions.sh"

log "============================================"
log " wav2vec-U  SETUP  (macOS Apple Silicon M2) "
log "============================================"
log "Project root : $INSTALL_ROOT"
log "Python venv  : $VENV_PATH"
log "KenLM bins   : $KENLM_ROOT"
log ""

# ---- 1. System dependencies via Homebrew ----
basic_dependencies

# ---- 2. Python venv (no CUDA, no pyenv forced install if already present) ----
setup_venv

# ---- 3. PyTorch (MPS-enabled Apple Silicon wheel) + other Python packages ----
install_pytorch_and_other_packages

# ---- 4. Fairseq in editable mode ----
install_fairseq

# ---- 5. Flashlight text + sequence (CPU-only build) ----
install_flashlight

# ---- 6. rVADfast (silence detection) ----
install_rVADfast

# ---- 7. KenLM (build from source if not already built) ----
install_kenlm

# ---- 8. Pre-trained wav2vec model ----
download_pretrained_model

# ---- 9. Language identification model (fasttext lid.176.bin) ----
download_languageIdentification_model

log ""
log "============================================"
log " SETUP COMPLETE"
log "============================================"
log ""
log "Next steps:"
log "  1. Download LibriSpeech text corpus (for prepare_text):"
log "     wget -P $INSTALL_ROOT/data/lm_data \\"
log "       https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"
log "     gunzip $INSTALL_ROOT/data/lm_data/librispeech-lm-norm.txt.gz"
log ""
log "  2. Run the pipeline:"
log "     bash run_wav2vec_mac.sh"
log ""
log "  3. Then train the GAN:"
log "     bash run_gans_mac.sh"
