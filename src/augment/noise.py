import numpy as np

from src.data_io.audio import load_audio, SAMPLE_RATE


def _snr_scale(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    """Return the scalar that scales noise to achieve the target SNR."""
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return 0.0
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    return float(np.sqrt(target_noise_power / noise_power))


def add_gaussian_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (dB)."""
    noise = np.random.randn(len(audio)).astype(np.float32)
    scale = _snr_scale(audio, noise, snr_db)
    return np.clip(audio + scale * noise, -1.0, 1.0)


def _pink_noise(n: int) -> np.ndarray:
    """Generate pink (1/f) noise via FFT spectral shaping."""
    white = np.fft.rfft(np.random.randn(n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    pink = white / np.sqrt(freqs)
    out = np.fft.irfft(pink, n=n).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-8)


def add_pink_noise(audio: np.ndarray, snr_db: float = 10.0) -> np.ndarray:
    """Add pink (1/f) noise — closer to real outdoor ambience than white noise."""
    noise = _pink_noise(len(audio))
    scale = _snr_scale(audio, noise, snr_db)
    return np.clip(audio + scale * noise, -1.0, 1.0)


def add_background(
    audio: np.ndarray,
    background_path: str,
    snr_db: float = 10.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix audio with a real background recording (e.g. a soundscape file).

    If the background is shorter than audio it is tiled; if longer a random
    offset is sampled so the same segment isn't always used.
    """
    bg = load_audio(background_path, target_sr=sr)

    if len(bg) < len(audio):
        repeats = int(np.ceil(len(audio) / len(bg)))
        bg = np.tile(bg, repeats)

    # random crop if background is longer
    if len(bg) > len(audio):
        offset = np.random.randint(0, len(bg) - len(audio))
        bg = bg[offset : offset + len(audio)]

    scale = _snr_scale(audio, bg, snr_db)
    return np.clip(audio + scale * bg, -1.0, 1.0)
