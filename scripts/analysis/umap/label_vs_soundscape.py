from pathlib import Path

import numpy as np

from src.models.birdnet import BirdNetEmbedder
from src.data_io.cache import load_or_embed
from src.visualize.umap_plot import compute_umap, plot_umap
from src.visualize.run_log import next_run_id, log_run

FOLDER_A = "data/birdclef-2026/train_audio"
FOLDER_B = "data/birdclef-2026/train_soundscapes"
BATCH_SIZE = 512   # audio chunks per GPU batch
CACHE_EVERY_N = 5  # save checkpoint every N batches

if __name__ == "__main__":
    embedder = BirdNetEmbedder()

    emb_a, _ = load_or_embed(FOLDER_A, embedder, batch_size=BATCH_SIZE, cache_every_n=CACHE_EVERY_N)
    emb_b, _ = load_or_embed(FOLDER_B, embedder, batch_size=BATCH_SIZE, cache_every_n=CACHE_EVERY_N)

    all_emb = np.concatenate([emb_a, emb_b])
    labels = [Path(FOLDER_A).name] * len(emb_a) + [Path(FOLDER_B).name] * len(emb_b)

    print(f"Total segments: {len(all_emb):,}")
    coords = compute_umap(all_emb)

    run_id = next_run_id()
    save_path = f"data/embeddings/umap_{run_id:03d}.png"
    plot_umap(coords, labels, title=f"UMAP run {run_id:03d}", save_path=save_path)
    log_run(run_id, FOLDER_A, FOLDER_B, save_path)