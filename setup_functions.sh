#!/bin/bash

# ==================== SETUP FUNCTIONS (macOS Apple Silicon) ====================
# Adapted from the original Google Colab / Linux / NVIDIA GPU version.
# This version targets macOS M2 arm64, using Homebrew and the MPS backend.

set -e
set -o pipefail

# Source utils for paths and helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

# ---- Ensure Homebrew is on PATH (Apple Silicon default: /opt/homebrew) ----
# The IDE / non-login shell may not have /opt/homebrew/bin in PATH.
BREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
export PATH="${BREW_PREFIX}/bin:${BREW_PREFIX}/sbin:${PATH}"

INSTALL_ROOT="$DIR_PATH"
PYTHON_VERSION="3.11"

# Python 3.11 is required — fairseq + hydra-core 1.0.7 use a dataclass pattern
# that Python 3.12 broke (mutable defaults). Python 3.11 is fully compatible.
# Homebrew 3.11 is preferred; falls back to 3.10 if needed.
if [ -x "${BREW_PREFIX}/bin/python3.11" ]; then
    PYTHON312="${BREW_PREFIX}/bin/python3.11"
elif [ -x "${BREW_PREFIX}/bin/python3.10" ]; then
    PYTHON312="${BREW_PREFIX}/bin/python3.10"
    PYTHON_VERSION="3.10"
elif [ -x "/usr/local/bin/python3.11" ]; then
    PYTHON312="/usr/local/bin/python3.11"
else
    echo "ERROR: Python 3.11 not found. Install with: brew install python@3.11"
    exit 1
fi

# Homebrew binary (explicit path as fallback)
BREW="${BREW_PREFIX}/bin/brew"


# ==================== HELPER ====================

log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

command_exists() {
    command -v "$1" > /dev/null 2>&1
}


# ==================== STEP 1: Basic System Dependencies (macOS) ====================

basic_dependencies() {
    log "Installing system dependencies via Homebrew..."

    if ! command_exists brew; then
        log "ERROR: Homebrew not found at $BREW_PREFIX. Install from https://brew.sh"
        exit 1
    fi

    # Core build tools + libraries needed for KenLM, flashlight, fairseq
    $BREW install cmake wget git ninja pkg-config zlib || true
    $BREW install boost eigen || true
    $BREW install sentencepiece || true

    # espeak-ng is used as the phonemizer (replaces espeak for macOS)
    if ! command_exists espeak-ng; then
        log "Installing espeak-ng..."
        $BREW install espeak-ng || log "WARNING: espeak-ng install failed. G2P phonemizer will be used instead."
    else
        log "espeak-ng already installed."
    fi

    log "System dependencies installed."
}


# ==================== STEP 2: GPU — macOS uses MPS, not CUDA ====================

cuda_installation() {
    log "INFO: CUDA is not available on Apple Silicon M2."
    log "INFO: PyTorch MPS (Metal Performance Shaders) will be used instead."
    log "INFO: Skipping CUDA installation."
}

gpu_drivers_installation() {
    log "INFO: NVIDIA GPU drivers are not applicable on macOS Apple Silicon."
    log "INFO: Metal GPU support is built into macOS — no drivers to install."
}


# ==================== STEP 3: Python Virtual Environment ====================

setup_venv() {
    log "Setting up Python virtual environment..."
    log "Using Python binary: $PYTHON312 ($( $PYTHON312 --version 2>&1))"
    local EXPECTED_MAJOR_MINOR
    EXPECTED_MAJOR_MINOR=$("$PYTHON312" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    # ---- Detect broken venv ----
    # The Colab-era venv has bin/python3 as a plain text file containing a
    # bare path (e.g. "/usr/bin/python3") rather than a proper symlink or
    # Python executable. We detect this by:
    #   1. Checking the 'executable' entry in pyvenv.cfg matches our expected Python
    #   2. Verifying bin/python3 is actually an ELF/Mach-O binary, not a text wrapper
    local VENV_FUNCTIONAL=false

    if [ -f "$VENV_PATH/pyvenv.cfg" ] && [ -f "$VENV_PATH/bin/python3" ]; then
        # Read the Python executable recorded in pyvenv.cfg
        local CFG_EXECUTABLE
        CFG_EXECUTABLE=$(grep '^executable' "$VENV_PATH/pyvenv.cfg" | awk '{print $3}')
        local CFG_VERSION
        CFG_VERSION=$(grep '^version' "$VENV_PATH/pyvenv.cfg" | awk '{print $3}' | cut -d. -f1-2)

        # Check if bin/python3 is a real binary (Mach-O) or a text wrapper
        local FILE_TYPE
        FILE_TYPE=$(file "$VENV_PATH/bin/python3" 2>/dev/null)
        local IS_BINARY=false
        if echo "$FILE_TYPE" | grep -qiE "Mach-O|ELF|symbolic link"; then
            IS_BINARY=true
        fi

        log "  pyvenv.cfg executable : $CFG_EXECUTABLE"
        log "  pyvenv.cfg version    : $CFG_VERSION"
        log "  bin/python3 type      : $FILE_TYPE"
        log "  Is binary/symlink     : $IS_BINARY"
        log "  Expected version      : $EXPECTED_MAJOR_MINOR"

        if [ "$IS_BINARY" = true ] && [ "$CFG_VERSION" = "$EXPECTED_MAJOR_MINOR" ]; then
            VENV_FUNCTIONAL=true
            log "Existing venv is functional"
        else
            log "Existing venv is broken (text wrapper or wrong Python version) — will rebuild"
        fi
    fi

    if [ "$VENV_FUNCTIONAL" = false ]; then
        if [ -d "$VENV_PATH" ]; then
            local BACKUP_DIR="${VENV_PATH}_backup_$(date +%Y%m%d%H%M%S)"
            log "Backing up broken venv to $BACKUP_DIR ..."
            mv "$VENV_PATH" "$BACKUP_DIR"
        fi
        log "Creating fresh venv with $PYTHON312 ..."
        "$PYTHON312" -m venv "$VENV_PATH"
        log "Fresh venv created."
    fi

    # Always use the venv's own pip — never the system pip (avoids PEP 668)
    source "$VENV_PATH/bin/activate"
    "$VENV_PATH/bin/pip3" install --upgrade pip
    log "Python virtual environment ready: $(python3 --version)"
}


# ==================== STEP 4: PyTorch for Apple Silicon (MPS) ====================

install_pytorch_and_other_packages() {
    log "Installing PyTorch and related packages for Apple Silicon..."
    source "$VENV_PATH/bin/activate"

    "$VENV_PATH/bin/pip3" install --upgrade pip

    # PyTorch 2.x ships MPS-enabled wheels for Apple Silicon by default
    # Use the standard index (NOT the CUDA cu* wheels)
    log "Installing PyTorch 2.3.0 (Apple Silicon MPS build)..."
    "$VENV_PATH/bin/pip3" install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

    # Verify MPS is available
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available:   {torch.backends.mps.is_available()}')
print(f'MPS built-in:    {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ MPS backend ready — GPU acceleration enabled via Metal')
else:
    print('⚠️  MPS not available — will use CPU')
"

    # Core numeric / audio packages
    "$VENV_PATH/bin/pip3" install "numpy<2" scipy tqdm sentencepiece soundfile librosa editdistance
    "$VENV_PATH/bin/pip3" install tensorboardX packaging npy-append-array h5py g2p_en

    # faiss-cpu (faiss-gpu not available on macOS arm64)
    "$VENV_PATH/bin/pip3" install faiss-cpu

    # ninja for C extensions
    "$VENV_PATH/bin/pip3" install ninja

    # NLTK data (for G2P phonemizer) — fix macOS SSL cert issue first
    # Python.org macOS installers don't use system certs; run the bundled installer
    CERT_CMD="/Applications/Python 3.12/Install Certificates.command"
    if [ -f "$CERT_CMD" ]; then
        bash "$CERT_CMD" > /dev/null 2>&1 || true
    else
        # Fallback: install certifi and point SSL to it
        "$VENV_PATH/bin/pip3" install certifi || true
        export SSL_CERT_FILE=$("$VENV_PATH/bin/python3" -c "import certifi; print(certifi.where())" 2>/dev/null) || true
    fi
    "$VENV_PATH/bin/python3" -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)" || true
    "$VENV_PATH/bin/python3" -c "import nltk; nltk.download('cmudict', quiet=True)" || true

    log "PyTorch and related packages installed successfully."
}


# ==================== STEP 5: Fairseq ====================

install_fairseq() {
    log "--- Installing fairseq ---"
    source "$VENV_PATH/bin/activate"

    "$VENV_PATH/bin/pip3" install "pip>=24.0"
    "$VENV_PATH/bin/pip3" install "cython<3.0"
    "$VENV_PATH/bin/pip3" install "setuptools>=40.8.0" wheel

    # ---- omegaconf 2.0.6 compatibility fix ----
    # pip>=24.1 rejects omegaconf 2.0.6 due to invalid PyYAML>=5.1.* metadata.
    # Fix: download the wheel, patch the metadata, repack with the correct name.
    log "Patching and installing omegaconf 2.0.6..."
    local OMEGACONF_WHL="/tmp/omegaconf-2.0.6-py3-none-any.whl"  # MUST keep full wheel name
    local OMEGACONF_PATCH="/tmp/omegaconf_patch"

    wget -q "https://files.pythonhosted.org/packages/d0/eb/9d63ce09dd8aa85767c65668d5414958ea29648a0eec80a4a7d311ec2684/omegaconf-2.0.6-py3-none-any.whl" \
        -O "$OMEGACONF_WHL" \
        || { log "[ERROR] Failed to download omegaconf wheel."; exit 1; }

    rm -rf "$OMEGACONF_PATCH" && mkdir -p "$OMEGACONF_PATCH"
    cd "$OMEGACONF_PATCH"
    unzip -q "$OMEGACONF_WHL"

    # macOS sed requires '' after -i
    sed -i '' 's/PyYAML (>=5\.1\.\*)/PyYAML (>=5.1)/' omegaconf-2.0.6.dist-info/METADATA

    # Repack — destination must use the full wheel filename for pip to accept it
    zip -q -r "$OMEGACONF_WHL" .
    cd "$INSTALL_ROOT"
    rm -rf "$OMEGACONF_PATCH"

    "$VENV_PATH/bin/pip3" install "$OMEGACONF_WHL" --no-deps \
        || { log "[ERROR] Failed to install patched omegaconf."; exit 1; }

    "$VENV_PATH/bin/pip3" install "PyYAML>=5.1"
    "$VENV_PATH/bin/pip3" install "hydra-core==1.0.7" --no-deps
    "$VENV_PATH/bin/pip3" install "antlr4-python3-runtime==4.8"
    # ---- end omegaconf fix ----

    "$VENV_PATH/bin/pip3" install "numpy<2"

    cd "$INSTALL_ROOT"

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists at $FAIRSEQ_ROOT"
        cd "$FAIRSEQ_ROOT"
    else
        log "Cloning fairseq repository..."
        git clone https://github.com/Ashesi-Org/fairseq_.git "$FAIRSEQ_ROOT" \
            || { log "[ERROR] Failed to clone fairseq."; exit 1; }
        cd "$FAIRSEQ_ROOT"
    fi

    log "Installing fairseq in editable mode (pure-Python, no C extensions)..."
    # fairseq's C++/CUDA extensions (libbleu, libnat, etc.) fail on macOS ARM64.
    # We patched setup.py to respect SKIP_EXTENSIONS=1 which skips all ext_modules.
    # wav2vec-U GAN training only needs the pure-Python fairseq components.
    cd "$FAIRSEQ_ROOT"

    local FAIRSEQ_BUILD_LOG="/tmp/fairseq_build.log"

    if SKIP_EXTENSIONS=1 CUDA_HOME="" \
        "$VENV_PATH/bin/pip3" install --editable ./ --no-deps --no-build-isolation \
        > "$FAIRSEQ_BUILD_LOG" 2>&1; then
        log "✅ fairseq installed successfully (pure-Python mode)."
        tail -3 "$FAIRSEQ_BUILD_LOG"
    else
        log "[ERROR] fairseq install failed even with SKIP_EXTENSIONS=1."
        tail -20 "$FAIRSEQ_BUILD_LOG"
        exit 1
    fi

    cd "$INSTALL_ROOT"
    "$VENV_PATH/bin/pip3" install sacrebleu bitarray tensorboardX

    local wav2vec_req_file="$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    if [ -f "$wav2vec_req_file" ]; then
        log "Installing wav2vec specific requirements..."
        "$VENV_PATH/bin/pip3" install -r "$wav2vec_req_file" \
            || log "[WARN] Some wav2vec requirements failed."
    fi

    log "fairseq installed successfully."
    deactivate
}


# ==================== STEP 6: rVADfast ====================

install_rVADfast() {
    log "Installing rVADfast..."
    cd "$INSTALL_ROOT"
    source "$VENV_PATH/bin/activate"

    if [ -d "$DIR_PATH/rVADfast" ]; then
        log "rVADfast already cloned. Trying to update..."
        cd "$DIR_PATH/rVADfast"
        git pull || log "[WARN] Could not pull rVADfast updates."
    else
        git clone https://github.com/zhenghuatan/rVADfast.git "$DIR_PATH/rVADfast"
        cd "$DIR_PATH/rVADfast"
    fi

    mkdir -p "$DIR_PATH/rVADfast/src"
    log "rVADfast ready."
}


# ==================== STEP 7: KenLM (macOS) ====================

install_kenlm() {
    log "Building KenLM..."
    cd "$INSTALL_ROOT"

    # macOS: use brew for build dependencies (not apt)
    $BREW install eigen boost || true

    if [ ! -d "$DIR_PATH/kenlm" ]; then
        git clone https://github.com/kpu/kenlm.git "$DIR_PATH/kenlm"
    fi

    cd "$DIR_PATH/kenlm"
    if [ -d "build" ] && [ -f "build/bin/lmplz" ]; then
        log "KenLM already built at $DIR_PATH/kenlm/build/bin"
    else
        mkdir -p build && cd build
        cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
                 -DBOOST_ROOT="${BREW_PREFIX}" \
                 -DEigen3_DIR="${BREW_PREFIX}/share/eigen3/cmake"
        make -j "$(sysctl -n hw.logicalcpu)"
        log "KenLM built successfully."
    fi

    source "$VENV_PATH/bin/activate"
    "$VENV_PATH/bin/pip3" install https://github.com/kpu/kenlm/archive/master.zip || \
        log "[WARN] KenLM Python bindings install failed (binaries still usable)."

    log "KenLM ready."
}


# ==================== STEP 8: Flashlight (CPU-only on macOS) ====================

install_flashlight() {
    log "--- Installing Flashlight (CPU-only for macOS) ---"
    source "$VENV_PATH/bin/activate"

    # Install both flashlight packages from PyPI (much simpler than building from source)
    log "Installing flashlight-text from PyPI..."
    "$VENV_PATH/bin/pip3" install flashlight-text \
        || { log "[ERROR] Failed to install flashlight-text."; exit 1; }

    log "Installing flashlight-sequence from PyPI..."
    "$VENV_PATH/bin/pip3" install flashlight-sequence \
        || { log "[WARN] flashlight-sequence PyPI install failed; trying from source..."
             # Fallback: build from source with cmake pre-step
             local FLASHLIGHT_SEQ_ROOT="$INSTALL_ROOT/sequence"
             if [ ! -d "$FLASHLIGHT_SEQ_ROOT" ]; then
                 git clone https://github.com/flashlight/sequence.git "$FLASHLIGHT_SEQ_ROOT"
             fi
             cd "$FLASHLIGHT_SEQ_ROOT"
             # Create missing version.py that setup.py expects
             mkdir -p bindings/python/flashlight/lib/sequence
             echo '__version__ = "0.0.0"' > bindings/python/flashlight/lib/sequence/version.py
             USE_CUDA=0 "$VENV_PATH/bin/pip3" install . -v \
                 || { log "[ERROR] Flashlight sequence build from source also failed."; exit 1; }
           }

    log "--- Flashlight installed ---"
}


# ==================== STEP 9: Download Pre-trained Model ====================

download_pretrained_model() {
    log "Checking for pre-trained wav2vec model..."
    mkdir -p "$INSTALL_ROOT/pre-trained"

    if [ -f "$INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt" ]; then
        log "Pre-trained model already exists at $INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt"
    else
        log "Downloading wav2vec_vox_new.pt (~3GB)..."
        wget -P "$INSTALL_ROOT/pre-trained" \
            https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt \
            || { log "[ERROR] Model download failed."; exit 1; }
        log "Model downloaded."
    fi
}


# ==================== STEP 10: Language Identification Model ====================

download_languageIdentification_model() {
    log "Checking for language identification model..."
    mkdir -p "$INSTALL_ROOT/lid_model"

    if [ -f "$INSTALL_ROOT/lid_model/lid.176.bin" ]; then
        log "LID model already exists at $INSTALL_ROOT/lid_model/lid.176.bin"
    else
        log "Downloading lid.176.bin..."
        wget -P "$INSTALL_ROOT/lid_model" \
            https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
            || { log "[ERROR] LID model download failed."; exit 1; }
    fi

    source "$VENV_PATH/bin/activate"
    "$VENV_PATH/bin/pip3" install fasttext-wheel || \
        "$VENV_PATH/bin/pip3" install fasttext || \
        log "[WARN] fasttext install failed. LID filtering will be skipped."
    log "LID model ready."
}