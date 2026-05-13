import numpy as np
import librosa

from src.data_io.audio import SAMPLE_RATE


def random_gain(audio: np.ndarray, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    """Scale amplitude by a random gain in [min_db, max_db]."""
    db = np.random.uniform(min_db, max_db)
    scale = 10 ** (db / 20)
    return np.clip(audio * scale, -1.0, 1.0)


def pitch_shift(audio: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float | None = None) -> np.ndarray:
    """Shift pitch by n_steps semitones (sampled uniformly in [-2, 2] if None)."""
    if n_steps is None:
        n_steps = np.random.uniform(-2.0, 2.0)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float | None = None) -> np.ndarray:
    """Stretch audio by rate (sampled uniformly in [0.9, 1.1] if None).

    rate > 1 speeds up, rate < 1 slows down. Output is trimmed/padded to
    the original length so downstream chunk sizes stay consistent.
    """
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    n = len(audio)
    if len(stretched) >= n:
        return stretched[:n]
    return np.pad(stretched, (0, n - len(stretched)))
