import shutil
import subprocess

import numpy as np
import birdnet

from src.data_io.audio import load_audio, chunk_audio, SAMPLE_RATE, CHUNK_DURATION

EMBEDDING_DIM = 1024


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


class BirdNetEmbedder:
    """Extracts BirdNET embeddings aligned to competition chunk windows.

    For a file of duration T and chunk_duration=5.0:
      - Returns shape (ceil(T/5), 1024)
      - Each row corresponds to one submission row ID (_5, _10, _15, …)

    BirdNET processes 3-second windows internally; embeddings are averaged
    across those internal windows to produce one vector per 5-second chunk.
    """

    def __init__(self, version: str = "2.4", backend: str = "pb"):
        # "pb" (ProtoBuf SavedModel) supports GPU; "tf" (TFLite) is CPU-only
        self.model = birdnet.load("acoustic", version, backend)
        # Use nvidia-smi instead of tf.config so we don't init the CUDA context
        # in the main process — BirdNET forks workers that need to init it themselves.
        self._device = "GPU" if _gpu_available() else "CPU"
        print(f"BirdNetEmbedder: device={self._device}")

    def get_chunks(
        self,
        filepath: str,
        chunk_duration: float = CHUNK_DURATION,
    ) -> list[np.ndarray]:
        """Load an audio file and return its raw audio chunks (no encoding)."""
        audio = load_audio(filepath)
        return chunk_audio(audio, sr=SAMPLE_RATE, chunk_duration=chunk_duration)

    def open_session(self, batch_size: int = 1):
        """Return a persistent encode_session context manager.

        Keep the session open across multiple encode_chunks_with_session calls
        so the model stays loaded in GPU memory between batches.
        """
        return self.model.encode_session(
            batch_size=batch_size,
            n_workers=1,
            prefetch_ratio=0,
            device=self._device,
        )

    def encode_chunks_with_session(
        self,
        session,
        chunks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Encode chunks using an already-open session (model stays resident)."""
        result = session.run_arrays([(chunk, SAMPLE_RATE) for chunk in chunks])
        return self._parse_result(result, len(chunks))

    def encode_chunks(self, chunks: list[np.ndarray], batch_size: int | None = None) -> list[np.ndarray]:
        """Encode chunks via a one-shot encode_arrays call (opens/closes session)."""
        result = self.model.encode_arrays(
            [(chunk, SAMPLE_RATE) for chunk in chunks],
            batch_size=batch_size if batch_size is not None else len(chunks),
            n_workers=1,
            prefetch_ratio=0,
            device=self._device,
        )
        return self._parse_result(result, len(chunks))

    def _parse_result(self, result, n_chunks: int) -> list[np.ndarray]:
        embeddings = []
        for i in range(n_chunks):
            segs = result.embeddings[i]
            mask = result.embeddings_masked[i]
            valid = ~mask.all(axis=1)
            embeddings.append(segs[valid] if valid.any() else segs)
        return embeddings

    def embed_file(
        self,
        filepath: str,
        chunk_duration: float = CHUNK_DURATION,
    ) -> list[np.ndarray]:
        """Return one embedding array per chunk window for an audio file."""
        chunks = self.get_chunks(filepath, chunk_duration=chunk_duration)
        return self.encode_chunks(chunks)
