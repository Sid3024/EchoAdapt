"""Step 1 of the search pipeline: create embeddings and prepare search data.

Run order:
  1. embed_soundscapes() — embed all soundscape files, cache to disk
  2. prepare_search_data() — subsample soundscape embeddings + select labelled file paths
  3. load_search_data()    — load prepared data for use during optimisation
"""

import json
from pathlib import Path

import numpy as np

from src.config.embedding import EmbeddingConfig
from src.config.search import SearchConfig
from src.data_io.cache import load_or_embed

_SEARCH_PATHS_FILE = "search_paths.json"
_SEARCH_SOUNDSCAPE_FILE = "search_soundscape.npy"


def embed_labelled(emb_config: EmbeddingConfig, embedder) -> np.ndarray:
    emb, _ = load_or_embed(
        emb_config.labelled_dir, embedder,
        batch_size=emb_config.batch_size, cache_dir=Path(emb_config.output_dir),
    )
    return emb


def embed_soundscapes(emb_config: EmbeddingConfig, embedder) -> np.ndarray:
    emb, _ = load_or_embed(
        emb_config.soundscape_dir, embedder,
        batch_size=emb_config.batch_size, cache_dir=Path(emb_config.output_dir),
    )
    return emb


def prepare_search_data(
    emb_config: EmbeddingConfig,
    search_config: SearchConfig,
) -> tuple[list[str], np.ndarray]:
    """Select labelled file paths and subsample soundscape embeddings.

    Saves a fixed selection to disk so every trial uses the same files.

    Returns:
        file_paths    list[str] — paths to labelled audio files for augmentation
        soundscape_emb  np.ndarray (subsample_n, 1024)
    """
    out = Path(emb_config.output_dir)
    paths_file = out / _SEARCH_PATHS_FILE
    sc_path = out / _SEARCH_SOUNDSCAPE_FILE

    if paths_file.exists() and sc_path.exists():
        print("Search data already prepared, loading from disk.")
        return json.loads(paths_file.read_text()), np.load(sc_path)

    rng = np.random.default_rng(search_config.seed)

    # --- soundscape subsample ---
    sc_emb = np.load(out / f"{Path(emb_config.soundscape_dir).name}.npy")
    idx = rng.choice(len(sc_emb), size=min(search_config.subsample_n, len(sc_emb)), replace=False)
    sc_sub = sc_emb[idx].astype(np.float32)

    # --- labelled file selection ---
    all_paths = sorted(Path(emb_config.labelled_dir).rglob("*.ogg"))
    chosen_idx = rng.choice(len(all_paths), size=min(search_config.n_audio_files, len(all_paths)), replace=False)
    chosen_paths = [str(all_paths[i]) for i in sorted(chosen_idx)]

    out.mkdir(parents=True, exist_ok=True)
    paths_file.write_text(json.dumps(chosen_paths))
    np.save(sc_path, sc_sub)

    print(f"Search data saved: {len(chosen_paths)} labelled files, {len(sc_sub)} soundscape embeddings")
    return chosen_paths, sc_sub


def load_search_data(emb_config: EmbeddingConfig) -> tuple[list[str], np.ndarray]:
    """Load pre-prepared search data from disk."""
    out = Path(emb_config.output_dir)
    file_paths = json.loads((out / _SEARCH_PATHS_FILE).read_text())
    soundscape_emb = np.load(out / _SEARCH_SOUNDSCAPE_FILE)
    return file_paths, soundscape_emb
