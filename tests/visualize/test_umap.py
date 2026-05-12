from pathlib import Path

import numpy as np

from src.models.birdnet import BirdNetEmbedder
from src.data_io.cache import load_or_embed
from src.visualize.umap_plot import compute_umap, plot_umap
from src.visualize.run_log import next_run_id, log_run

FOLDER_A = "data/birdclef-2026/train_audio/23150"
FOLDER_B = "data/birdclef-2026/train_audio/23176"

# for actual umap use:
# FOLDER_A = "data/birdclef-2026/train_audio"
# FOLDER_B = "data/birdclef-2026/train_soundscapes"

if __name__ == "__main__":
    embedder = BirdNetEmbedder()

    emb_a, _ = load_or_embed(FOLDER_A, embedder)
    emb_b, _ = load_or_embed(FOLDER_B, embedder)

    all_emb = np.concatenate([emb_a, emb_b])
    labels = [Path(FOLDER_A).name] * len(emb_a) + [Path(FOLDER_B).name] * len(emb_b)

    print(f"Total segments: {len(all_emb):,}")
    coords = compute_umap(all_emb)

    run_id = next_run_id()
    save_path = f"data/embeddings/umap_{run_id:03d}.png"
    plot_umap(coords, labels, title=f"UMAP run {run_id:03d}", save_path=save_path)
    log_run(run_id, FOLDER_A, FOLDER_B, save_path)
