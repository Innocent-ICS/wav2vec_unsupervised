#!/bin/bash
# finish_prepare_audio.sh
# Runs all remaining prepare_audio.sh steps after feature extraction:
#   C. Apply clustering labels (all splits)
#   D. PCA on train features
#   E. Apply PCA (all splits)
#   F. Mean pooling
#   G. Pooled feature normalization
# Then marks prepare_audio as COMPLETED and runs prepare_text.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"

VENV_PY="$(pwd)/venv/bin/python3"
FAIRSEQ="$(pwd)/fairseq_"
CLUSTER="$(pwd)/data/clustering/librispeech"
MODEL="$(pwd)/pre-trained/wav2vec_vox_new.pt"
DIM=512
SPLITS="train valid test"

source utils.sh   # for log(), mark_completed(), etc.

echo ""
echo "=== STEP C: Apply CLUS128 labels to all splits ==="
for split in $SPLITS; do
    echo "  Applying clustering → $split..."
    $VENV_PY "$FAIRSEQ/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py" \
        "$CLUSTER" \
        --checkpoint "$MODEL" \
        --path "$CLUSTER/CLUS128" \
        --split "$split" 2>&1 | tail -2
    echo "  ✅ $split done"
done

echo ""
echo "=== STEP D: PCA on train.npy (dim=$DIM) ==="
$VENV_PY "$FAIRSEQ/examples/wav2vec/unsupervised/scripts/pca.py" \
    "$CLUSTER/train.npy" \
    --output "$CLUSTER/pca" \
    --dim $DIM 2>&1 | tail -2
echo "PCA model: $(ls $CLUSTER/pca/)"

echo ""
echo "=== STEP E: Apply PCA to all splits ==="
for split in $SPLITS; do
    echo "  Applying PCA → $split..."
    $VENV_PY "$FAIRSEQ/examples/wav2vec/unsupervised/scripts/apply_pca.py" \
        "$CLUSTER" \
        --split "$split" \
        --save-dir "$CLUSTER/precompute_pca${DIM}" \
        --pca-path "$CLUSTER/pca/${DIM}_pca" \
        --batch-size 32 2>&1 | tail -2
    echo "  ✅ $split done"
done

echo ""
echo "=== STEP F: Apply mean pooling with CLUS128 ==="
for split in $SPLITS; do
    echo "  Mean pooling → $split..."
    $VENV_PY "$FAIRSEQ/examples/wav2vec/unsupervised/scripts/mean_pool.py" \
        "$CLUSTER/precompute_pca${DIM}" \
        --split "$split" \
        --save-dir "$CLUSTER/precompute_pca${DIM}_cls128_mean" \
        --pooling mean 2>&1 | tail -2
    echo "  ✅ $split done"
done

echo ""
echo "=== STEP G: Apply pooled PCA normalization ==="
for split in $SPLITS; do
    echo "  Pooled norm → $split..."
    $VENV_PY "$FAIRSEQ/examples/wav2vec/unsupervised/scripts/apply_pca.py" \
        "$CLUSTER/precompute_pca${DIM}_cls128_mean" \
        --split "$split" \
        --save-dir "$CLUSTER/precompute_pca${DIM}_cls128_mean_pooled" \
        --pca-path "$CLUSTER/pca/${DIM}_pca" \
        --batch-size 32 2>&1 | tail -2
    echo "  ✅ $split done"
done

echo ""
echo "=== Marking prepare_audio COMPLETED ==="
# Update the checkpoint file
CHECKPOINT_FILE="$(pwd)/data/checkpoints/librispeech/progress.checkpoint"
# Remove IN_PROGRESS entry, add COMPLETED
grep -v "^prepare_audio:" "$CHECKPOINT_FILE" > /tmp/checkpoint_tmp.txt || true
echo "prepare_audio:COMPLETED" >> /tmp/checkpoint_tmp.txt
mv /tmp/checkpoint_tmp.txt "$CHECKPOINT_FILE"
echo "Checkpoint updated"

echo ""
echo "=== Final audio prep output ==="
ls -lh "$CLUSTER/precompute_pca${DIM}_cls128_mean_pooled/" 2>/dev/null | head -10

echo ""
echo "================================================"
echo " prepare_audio COMPLETE ✅"
echo " Next: run prepare_text (G2P + KenLM)"
echo "================================================"
