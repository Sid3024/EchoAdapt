import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.birdnet import BirdNetEmbedder

CACHE_DIR = Path("data/embeddings")


def load_or_embed(
    folder: str,
    embedder: "BirdNetEmbedder",
    batch_size: int = 64,
    cache_every_n: int = 10,
) -> tuple[np.ndarray, list[dict]]:
    """Return embeddings for all .ogg files under folder, using disk cache if available."""
    name = Path(folder).name
    emb_path = CACHE_DIR / f"{name}.npy"
    meta_path = CACHE_DIR / f"{name}_meta.json"

    if emb_path.exists() and meta_path.exists():
        print(f"Loading cached embeddings for '{name}'...")
        return np.load(emb_path), json.loads(meta_path.read_text())

    from src.data_io.folder import embed_folder
    print(f"Embedding '{name}'...")
    emb, meta = embed_folder(folder, embedder, batch_size=batch_size, cache_every_n=cache_every_n)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    meta_path.write_text(json.dumps(meta))
    print(f"Cached to {emb_path}")
    return emb, meta
