import numpy as np
import librosa

SAMPLE_RATE = 48000
CHUNK_DURATION = 5.0  # seconds — matches BirdCLEF submission row IDs (_5, _10, …)


def load_audio(filepath: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as a mono float32 array at target_sr."""
    audio, _ = librosa.load(filepath, sr=target_sr, mono=True)
    return audio


def chunk_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    chunk_duration: float = CHUNK_DURATION,
) -> list[np.ndarray]:
    """Split audio into fixed-length chunks, zero-padding the final chunk."""
    chunk_len = int(sr * chunk_duration)
    chunks = []
    for start in range(0, max(len(audio), 1), chunk_len):
        chunk = audio[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk.astype(np.float32))
    return chunks
