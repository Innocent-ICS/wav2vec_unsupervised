#!/usr/bin/env python3
"""
direct_cluster.py — Direct faiss k-means on pre-extracted train.npy
Bypasses wav2vec_cluster_faiss.py's audio re-extraction.
Replicates the CLUS128 output format expected by wav2vec_apply_cluster_faiss.py.

Usage:
    python3 direct_cluster.py \
        --npy data/clustering/librispeech/train.npy \
        --save-dir data/clustering/librispeech/CLUS128 \
        --n-clusters 128 \
        --sample-pct 0.5 \
        --seed 42
"""

import argparse
import os
import random
import numpy as np
import faiss


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="Path to pre-extracted train.npy")
    p.add_argument("--save-dir", required=True, help="Directory to save cluster model")
    p.add_argument("--n-clusters", type=int, default=128)
    p.add_argument("--sample-pct", type=float, default=0.5,
                   help="Fraction of frames to subsample for clustering")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--niter", type=int, default=40, help="Number of k-means iterations")
    return p


def main():
    args = get_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading features from {args.npy} ...")
    feats = np.load(args.npy, mmap_mode="r")
    print(f"  Shape: {feats.shape}, dtype: {feats.dtype}")

    n_total = feats.shape[0]
    dim = feats.shape[1]

    # Subsample frames
    random.seed(args.seed)
    np.random.seed(args.seed)
    n_sample = max(args.n_clusters, int(n_total * args.sample_pct))
    n_sample = min(n_sample, n_total)
    idx = np.random.choice(n_total, n_sample, replace=False)
    sample = feats[idx].astype(np.float32)
    print(f"  Sampled {n_sample} / {n_total} frames for clustering")

    print(f"Running faiss k-means: k={args.n_clusters}, dim={dim}, niter={args.niter} ...")
    kmeans = faiss.Kmeans(
        dim,
        args.n_clusters,
        niter=args.niter,
        verbose=True,
        gpu=False,
        seed=args.seed,
    )
    kmeans.train(sample)

    # Save centroid matrix as npy (wav2vec_apply_cluster_faiss.py loads this)
    centroids = kmeans.centroids
    out_path = os.path.join(args.save_dir, "centroids.npy")
    np.save(out_path, centroids)
    print(f"Saved centroids → {out_path}  shape={centroids.shape}")

    # Also save the faiss index (for compatibility with apply script)
    index = faiss.IndexFlatL2(dim)
    index.add(centroids)
    idx_path = os.path.join(args.save_dir, "index.faiss")
    faiss.write_index(index, idx_path)
    print(f"Saved faiss index → {idx_path}")

    print("Done!")


if __name__ == "__main__":
    main()
