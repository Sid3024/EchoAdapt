"""Sanity check: apply noise augmentations and inspect output statistics."""

import numpy as np
from src.augment.noise import add_gaussian_noise, add_pink_noise

SR = 48000
DURATION = 3.0
# synthetic signal: 1kHz sine wave (representative of a bird call in shape)
t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
audio = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

print(f"Original  | mean={audio.mean():.4f}  std={audio.std():.4f}  max={np.abs(audio).max():.4f}")

for snr in [5, 10, 20]:
    noisy = add_gaussian_noise(audio, snr_db=snr)
    print(f"Gaussian SNR={snr:>2}dB | mean={noisy.mean():.4f}  std={noisy.std():.4f}  max={np.abs(noisy).max():.4f}  clipped={np.sum(np.abs(noisy) >= 1.0)}")

print()
for snr in [5, 10, 20]:
    noisy = add_pink_noise(audio, snr_db=snr)
    print(f"Pink     SNR={snr:>2}dB | mean={noisy.mean():.4f}  std={noisy.std():.4f}  max={np.abs(noisy).max():.4f}  clipped={np.sum(np.abs(noisy) >= 1.0)}")
