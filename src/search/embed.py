"""Step 1 of the search pipeline: create embeddings and prepare search data.

Run order:
  1. embed_labelled()    — embed all train_audio files, cache to disk
  2. embed_soundscapes() — embed all soundscape files, cache to disk
  3. prepare_search_data() — subsample raw audio chunks + soundscape embeddings
                             and save a fixed dataset for use during optimisation
"""

from pathlib import Path

import numpy as np

from src.config.embedding import EmbeddingConfig
from src.config.search import SearchConfig
from src.data_io.audio import load_audio, chunk_audio, SAMPLE_RATE, CHUNK_DURATION
from src.data_io.cache import load_or_embed

_SEARCH_AUDIO_FILE = "search_audio.npy"
_SEARCH_SOUNDSCAPE_FILE = "search_soundscape.npy"


def embed_labelled(emb_config: EmbeddingConfig, embedder) -> np.ndarray:
    """Embed all labelled train_audio files and cache to disk.

    Returns:
        np.ndarray of shape (total_segments, 1024)
    """
    emb, _ = load_or_embed(emb_config.labelled_dir, embedder, batch_size=emb_config.batch_size)
    return emb


def embed_soundscapes(emb_config: EmbeddingConfig, embedder) -> np.ndarray:
    """Embed all soundscape files and cache to disk.

    Returns:
        np.ndarray of shape (total_segments, 1024)
    """
    emb, _ = load_or_embed(emb_config.soundscape_dir, embedder, batch_size=emb_config.batch_size)
    return emb


def prepare_search_data(
    emb_config: EmbeddingConfig,
    search_config: SearchConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and save a fixed subsample used throughout the optimisation search.

    Loads raw labelled audio chunks (to be augmented per trial) and subsamples
    soundscape embeddings (fixed reference, never augmented).

    Returns:
        labelled_chunks  np.ndarray (n_chunks, chunk_samples) — raw audio
        soundscape_emb   np.ndarray (subsample_n, 1024)
    """
    out = Path(emb_config.output_dir)
    audio_path = out / _SEARCH_AUDIO_FILE
    sc_path = out / _SEARCH_SOUNDSCAPE_FILE

    if audio_path.exists() and sc_path.exists():
        print("Search data already prepared, loading from disk.")
        return np.load(audio_path), np.load(sc_path)

    rng = np.random.default_rng(search_config.seed)

    # --- soundscape subsample ---
    sc_emb = np.load(out / f"{Path(emb_config.soundscape_dir).name}.npy")
    idx = rng.choice(len(sc_emb), size=min(search_config.subsample_n, len(sc_emb)), replace=False)
    sc_sub = sc_emb[idx].astype(np.float32)

    # --- labelled raw audio subsample ---
    paths = sorted(Path(emb_config.labelled_dir).rglob("*.ogg"))
    chosen_idx = rng.choice(len(paths), size=min(search_config.n_audio_files, len(paths)), replace=False)

    chunk_len = int(SAMPLE_RATE * CHUNK_DURATION)
    all_chunks: list[np.ndarray] = []
    for i in chosen_idx:
        audio = load_audio(str(paths[i]))
        all_chunks.extend(chunk_audio(audio, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION))

    # all chunks are same length (chunk_audio zero-pads), so we can stack
    labelled_chunks = np.stack(all_chunks).astype(np.float32)  # (n_chunks, chunk_len)

    out.mkdir(parents=True, exist_ok=True)
    np.save(audio_path, labelled_chunks)
    np.save(sc_path, sc_sub)

    print(f"Search data saved: {len(labelled_chunks)} audio chunks, {len(sc_sub)} soundscape embeddings")
    return labelled_chunks, sc_sub


def load_search_data(emb_config: EmbeddingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-prepared search data from disk."""
    out = Path(emb_config.output_dir)
    labelled_chunks = np.load(out / _SEARCH_AUDIO_FILE)
    soundscape_emb = np.load(out / _SEARCH_SOUNDSCAPE_FILE)
    return labelled_chunks, soundscape_emb
