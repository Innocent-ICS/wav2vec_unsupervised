#!/usr/bin/env python3
"""
fast_prepare_audio.py
=====================
Completes all remaining prepare_audio.sh steps using pre-extracted .npy files.
Bypasses audio re-extraction entirely — runs in minutes instead of hours.

Steps:
  1. Cluster train.npy with faiss k-means → CLUS128/centroids.npy
  2. Apply clustering to all splits → CLUS128/{split}.src + .tsv
  3. PCA on train.npy → pca/{dim}_pca.npy
  4. Apply PCA to all splits → precompute_pca{dim}/{split}.npy + .lengths + .tsv
  5. Mean-pool PCA features per utterance → precompute_pca{dim}_cls128_mean/
  6. Apply PCA to mean-pooled → precompute_pca{dim}_cls128_mean_pooled/
"""

import argparse
import os
import os.path as osp
import sys
import numpy as np
import faiss

DIM      = 512
N_CLUS   = 128
SAMPLE   = 0.5      # fraction of frames for k-means
NITER    = 40
SEED     = 42
SPLITS   = ["train", "valid", "test"]


# ─────────────────────────────────────────────────────────────────────────────
def load_npy_and_lengths(base_dir, split):
    """Load stacked feature array and per-utterance lengths."""
    npy_path = osp.join(base_dir, f"{split}.npy")
    len_path = osp.join(base_dir, f"{split}.lengths")
    feats   = np.load(npy_path, mmap_mode="r").astype(np.float32)
    lengths = [int(l.strip()) for l in open(len_path) if l.strip()]
    assert sum(lengths) == len(feats), \
        f"Mismatch: sum(lengths)={sum(lengths)} vs npy rows={len(feats)}"
    return feats, lengths


def split_by_lengths(feats, lengths):
    """Yield per-utterance sub-arrays."""
    idx = 0
    for l in lengths:
        yield feats[idx: idx + l]
        idx += l


def load_tsv(base_dir, split):
    """Return (root, list-of-tab-lines) from {split}.tsv."""
    with open(osp.join(base_dir, f"{split}.tsv")) as f:
        lines = f.read().splitlines()
    root = lines[0]
    rows = [l for l in lines[1:] if l.strip()]
    return root, rows


def save_npy_and_lengths(feats_list, save_path_npy, save_path_len):
    """Stack list of arrays, save .npy and .lengths."""
    stacked = np.vstack(feats_list).astype(np.float32)
    np.save(save_path_npy, stacked)
    with open(save_path_len, "w") as f:
        for arr in feats_list:
            f.write(f"{len(arr)}\n")
    return stacked


# ─────────────────────────────────────────────────────────────────────────────
def step_cluster(cluster_dir, sample_pct=SAMPLE, n_clus=N_CLUS, niter=NITER):
    clus128_dir = osp.join(cluster_dir, "CLUS128")
    os.makedirs(clus128_dir, exist_ok=True)
    centroids_path = osp.join(clus128_dir, "centroids.npy")

    if osp.exists(centroids_path):
        print(f"[STEP 1] CLUS128 centroids already exist, skipping k-means")
        return np.load(centroids_path)

    print(f"\n[STEP 1] Running faiss k-means on train.npy ...")
    feats, lengths = load_npy_and_lengths(cluster_dir, "train")
    n_total = len(feats)
    n_sample = max(n_clus * 10, int(n_total * sample_pct))
    n_sample = min(n_sample, n_total)
    np.random.seed(SEED)
    idx = np.random.choice(n_total, n_sample, replace=False)
    sample = feats[idx].copy()
    print(f"  Sampled {n_sample}/{n_total} frames (dim={feats.shape[1]})")

    km = faiss.Kmeans(feats.shape[1], n_clus, niter=niter, verbose=True,
                      gpu=False, seed=SEED)
    km.train(sample)
    centroids = km.centroids
    np.save(centroids_path, centroids)
    print(f"  Saved centroids → {centroids_path}  shape={centroids.shape}")
    return centroids


def step_apply_cluster(cluster_dir, centroids):
    """Write CLUS128/{split}.src and CLUS128/{split}.tsv for all splits."""
    clus128_dir = osp.join(cluster_dir, "CLUS128")
    print(f"\n[STEP 2] Applying clustering to all splits ...")
    dim = centroids.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(centroids)

    for split in SPLITS:
        src_path = osp.join(clus128_dir, f"{split}.src")
        tsv_path = osp.join(clus128_dir, f"{split}.tsv")
        if osp.exists(src_path):
            print(f"  {split}: already exists, skipping")
            continue
        print(f"  Processing {split} ...")
        feats, lengths = load_npy_and_lengths(cluster_dir, split)
        root, rows     = load_tsv(cluster_dir, split)

        with open(src_path, "w") as fp, open(tsv_path, "w") as tp:
            print(root, file=tp)
            for utt_feats, row in zip(split_by_lengths(feats, lengths), rows):
                _, z = index.search(utt_feats.copy(), 1)
                print(" ".join(str(x.item()) for x in z.flatten()), file=fp)
                print(row, file=tp)
        print(f"  ✅ {split}.src written ({len(lengths)} utterances)")


def step_pca(cluster_dir, dim=DIM):
    pca_dir = osp.join(cluster_dir, "pca")
    os.makedirs(pca_dir, exist_ok=True)
    pca_path = osp.join(pca_dir, f"{dim}_pca.npy")

    if osp.exists(pca_path):
        print(f"\n[STEP 3] PCA already computed, loading ...")
        return np.load(pca_path)

    print(f"\n[STEP 3] Fitting PCA (dim={dim}) on train.npy ...")
    feats, _ = load_npy_and_lengths(cluster_dir, "train")
    n, d = feats.shape
    print(f"  Input shape: {feats.shape}")

    # Subtract mean
    mean = feats.mean(axis=0)
    centered = feats - mean

    # SVD (economy)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_matrix = Vt[:dim].T  # shape (d, dim)

    # Pack as dict: {matrix, mean}
    pca = {"A": pca_matrix.astype(np.float32), "mean": mean.astype(np.float32)}
    np.save(pca_path, pca)
    print(f"  Saved PCA → {pca_path}  matrix shape={pca_matrix.shape}")
    return pca


def apply_pca_transform(feats, pca):
    """Apply PCA: center then project."""
    if isinstance(pca, dict):
        A, mean = pca["A"], pca["mean"]
    else:
        # loaded as 0-d object array
        d = pca.item()
        A, mean = d["A"], d["mean"]
    return (feats - mean) @ A


def step_apply_pca(cluster_dir, pca, dim=DIM):
    out_dir = osp.join(cluster_dir, f"precompute_pca{dim}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[STEP 4] Applying PCA to all splits → {out_dir} ...")

    for split in SPLITS:
        if osp.exists(osp.join(out_dir, f"{split}.npy")):
            print(f"  {split}: already exists, skipping")
            continue
        print(f"  Processing {split} ...")
        feats, lengths = load_npy_and_lengths(cluster_dir, split)
        root, rows     = load_tsv(cluster_dir, split)

        projected = apply_pca_transform(feats, pca)
        utts = list(split_by_lengths(projected, lengths))
        save_npy_and_lengths(utts,
                             osp.join(out_dir, f"{split}.npy"),
                             osp.join(out_dir, f"{split}.lengths"))
        # Copy tsv
        with open(osp.join(out_dir, f"{split}.tsv"), "w") as f:
            f.write(root + "\n")
            f.write("\n".join(rows) + "\n")
        print(f"  ✅ {split}: {len(utts)} utterances, shape={projected.shape}")


def step_mean_pool(cluster_dir, dim=DIM):
    in_dir  = osp.join(cluster_dir, f"precompute_pca{dim}")
    out_dir = osp.join(cluster_dir, f"precompute_pca{dim}_cls128_mean")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[STEP 5] Mean-pooling PCA features per utterance ...")

    for split in SPLITS:
        if osp.exists(osp.join(out_dir, f"{split}.npy")):
            print(f"  {split}: already exists, skipping")
            continue
        feats, lengths = load_npy_and_lengths(in_dir, split)
        root, rows     = load_tsv(in_dir, split)

        pooled = [utt.mean(axis=0, keepdims=True)
                  for utt in split_by_lengths(feats, lengths)]
        save_npy_and_lengths(pooled,
                             osp.join(out_dir, f"{split}.npy"),
                             osp.join(out_dir, f"{split}.lengths"))
        with open(osp.join(out_dir, f"{split}.tsv"), "w") as f:
            f.write(root + "\n")
            f.write("\n".join(rows) + "\n")
        print(f"  ✅ {split}: pooled to {len(pooled)} vectors")


def step_apply_pca_pooled(cluster_dir, pca, dim=DIM):
    in_dir  = osp.join(cluster_dir, f"precompute_pca{dim}_cls128_mean")
    out_dir = osp.join(cluster_dir, f"precompute_pca{dim}_cls128_mean_pooled")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[STEP 6] Applying PCA to mean-pooled features ...")

    for split in SPLITS:
        if osp.exists(osp.join(out_dir, f"{split}.npy")):
            print(f"  {split}: already exists, skipping")
            continue
        feats, lengths = load_npy_and_lengths(in_dir, split)
        root, rows     = load_tsv(in_dir, split)

        projected = apply_pca_transform(feats, pca)
        utts = list(split_by_lengths(projected, lengths))
        save_npy_and_lengths(utts,
                             osp.join(out_dir, f"{split}.npy"),
                             osp.join(out_dir, f"{split}.lengths"))
        with open(osp.join(out_dir, f"{split}.tsv"), "w") as f:
            f.write(root + "\n")
            f.write("\n".join(rows) + "\n")
        print(f"  ✅ {split}: {len(utts)} pooled+projected utterances")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-dir", required=True,
                        help="e.g. data/clustering/librispeech")
    parser.add_argument("--dim",      type=int, default=DIM)
    parser.add_argument("--n-clus",   type=int, default=N_CLUS)
    parser.add_argument("--sample-pct", type=float, default=SAMPLE)
    parser.add_argument("--niter",    type=int, default=NITER)
    args = parser.parse_args()

    cluster_dir = args.cluster_dir

    print("=" * 60)
    print(f"fast_prepare_audio.py — cluster_dir: {cluster_dir}")
    print("=" * 60)

    centroids = step_cluster(cluster_dir, args.sample_pct, args.n_clus, args.niter)
    step_apply_cluster(cluster_dir, centroids)
    pca = step_pca(cluster_dir, args.dim)
    step_apply_pca(cluster_dir, pca, args.dim)
    step_mean_pool(cluster_dir, args.dim)
    step_apply_pca_pooled(cluster_dir, pca, args.dim)

    print("\n" + "=" * 60)
    print("✅  All prepare_audio steps COMPLETE")
    print(f"    GAN input: {cluster_dir}/precompute_pca{args.dim}_cls128_mean_pooled/")
    print("=" * 60)


if __name__ == "__main__":
    main()
