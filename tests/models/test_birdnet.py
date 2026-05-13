"""Sanity check: embed one file and inspect output shape and values.

Requires BirdNET model (GPU recommended). Run with:
    python tests/models/test_birdnet.py
"""

import numpy as np

FILE = "data/birdclef-2026/train_audio/22956/XC900618.ogg"

if __name__ == "__main__":
    from src.models.birdnet import BirdNetEmbedder

    embedder = BirdNetEmbedder()
    chunks = embedder.embed_file(FILE)

    print(f"File: {FILE}")
    print(f"Chunks: {len(chunks)}")
    for i, emb in enumerate(chunks):
        print(f"  chunk {i}: shape={emb.shape}  mean={emb.mean():.4f}  std={emb.std():.4f}  min={emb.min():.4f}  max={emb.max():.4f}")

    all_emb = np.vstack(chunks)
    print(f"\nAll segments stacked: {all_emb.shape}")
    print(f"Any NaN: {np.isnan(all_emb).any()}")
    print(f"Any Inf: {np.isinf(all_emb).any()}")
