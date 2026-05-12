from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.birdnet import BirdNetEmbedder


def embed_folder(
    folder: str,
    embedder: BirdNetEmbedder,
) -> tuple[np.ndarray, list[dict]]:
    """Embed every .ogg file found recursively under folder.

    Each BirdNET internal 3-second segment is one row — segments are NOT
    averaged so that per-window classification and max-pooling works downstream.

    Works for both folder layouts:
      - train_audio/{species_id}/*.ogg  (nested by species)
      - train_soundscapes/*.ogg         (flat)

    Returns:
        embeddings  np.ndarray of shape (total_segments, 1024)
        meta        list[dict] with keys: filepath, chunk_idx, seg_idx
    """
    paths = sorted(Path(folder).rglob("*.ogg"))
    if not paths:
        raise FileNotFoundError(f"No .ogg files found under {folder!r}")

    all_embeddings: list[np.ndarray] = []
    all_meta: list[dict] = []

    for i, path in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] {path}")
        chunks = embedder.embed_file(str(path))  # list[(n_segs, 1024)]
        for chunk_idx, segs in enumerate(chunks):
            for seg_idx in range(len(segs)):
                all_embeddings.append(segs[seg_idx])
                all_meta.append(
                    {
                        "filepath": str(path),
                        "chunk_idx": chunk_idx,
                        "seg_idx": seg_idx,
                    }
                )

    return np.stack(all_embeddings), all_meta
