import numpy as np
import birdnet
import tensorflow as tf

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
        gpus = tf.config.list_physical_devices("GPU")
        device = f"GPU ({gpus[0].name})" if gpus else "CPU"
        print(f"BirdNetEmbedder: running on {device}")

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

        # batch all chunks in one GPU call instead of N separate calls
        result = self.model.encode_arrays(*[(chunk, SAMPLE_RATE) for chunk in chunks])

        embeddings = []
        for i in range(len(chunks)):
            segs = result.embeddings[i]         # (n_segs, 1024)
            mask = result.embeddings_masked[i]  # (n_segs, 1024) bool
            valid = ~mask.all(axis=1)
            embeddings.append(segs[valid] if valid.any() else segs)

        return embeddings
