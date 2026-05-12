import numpy as np
import birdnet

from src.data_io.audio import load_audio, chunk_audio, SAMPLE_RATE, CHUNK_DURATION

EMBEDDING_DIM = 1024


class BirdNetEmbedder:
    """Extracts BirdNET embeddings aligned to competition chunk windows.

    For a file of duration T and chunk_duration=5.0:
      - Returns shape (ceil(T/5), 1024)
      - Each row corresponds to one submission row ID (_5, _10, _15, …)

    BirdNET processes 3-second windows internally; embeddings are averaged
    across those internal windows to produce one vector per 5-second chunk.
    """

    def __init__(self, version: str = "2.4", backend: str = "tf"):
        # backend: "tf" (TensorFlow SavedModel, supports GPU) or "pb" (ProtoBuf)
        self.model = birdnet.load("acoustic", version, backend)

    def embed_file(
        self,
        filepath: str,
        chunk_duration: float = CHUNK_DURATION,
    ) -> np.ndarray:
        """Return one embedding per chunk window for an audio file.

        Args:
            filepath:       Path to the audio file.
            chunk_duration: Window size in seconds (default 5.0 to match
                            BirdCLEF submission row IDs).

        Returns:
            np.ndarray of shape (n_chunks, 1024).
        """
        audio = load_audio(filepath)
        chunks = chunk_audio(audio, sr=SAMPLE_RATE, chunk_duration=chunk_duration)

        embeddings = []
        for chunk in chunks:
            # encode_arrays takes (audio_array, sample_rate) — no temp files needed
            result = self.model.encode_arrays((chunk, SAMPLE_RATE))
            # result.embeddings shape: (n_inputs=1, n_internal_segments, 1024)
            # result.embeddings_masked: True where a value is padding/invalid
            segs = result.embeddings[0]        # (n_segs, 1024)
            mask = result.embeddings_masked[0] # (n_segs, 1024) bool
            valid = ~mask.all(axis=1)          # (n_segs,) — drop fully-padded segments
            valid_segs = segs[valid] if valid.any() else segs
            embeddings.append(valid_segs)

        # (n_chunks, n_internal_segs, 1024) — caller max-pools after classification
        return embeddings
