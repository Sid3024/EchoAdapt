import numpy as np


def overlay(
    audio: np.ndarray,
    other: np.ndarray,
    snr_db: float = 5.0,
) -> np.ndarray:
    """Overlay a second bird recording on top of audio at a given SNR.

    Simulates multiple species calling simultaneously, as in soundscapes.
    other is trimmed/tiled to match the length of audio.
    """
    n = len(audio)
    if len(other) < n:
        repeats = int(np.ceil(n / len(other)))
        other = np.tile(other, repeats)
    if len(other) > n:
        offset = np.random.randint(0, len(other) - n)
        other = other[offset : offset + n]

    sig_power = np.mean(audio ** 2)
    other_power = np.mean(other ** 2)
    if other_power > 0:
        target = sig_power / (10 ** (snr_db / 10))
        scale = float(np.sqrt(target / other_power))
    else:
        scale = 0.0

    return np.clip(audio + scale * other, -1.0, 1.0)


def mixup(
    audio1: np.ndarray,
    label1: np.ndarray,
    audio2: np.ndarray,
    label2: np.ndarray,
    alpha: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Mixup: linear interpolation of two examples and their labels.

    lambda is sampled from Beta(alpha, alpha). Both arrays must be the same
    length; labels must be 1-D float arrays (e.g. one-hot or soft).
    """
    lam = float(np.random.beta(alpha, alpha))
    n = min(len(audio1), len(audio2))
    mixed_audio = lam * audio1[:n] + (1 - lam) * audio2[:n]
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_audio.astype(np.float32), mixed_label.astype(np.float32)
