"""Sanity check: overlay and mixup."""

import numpy as np
from src.augment.mix import overlay, mixup

SR = 48000
DURATION = 3.0
n = int(SR * DURATION)

# two distinct signals
bird_a = (0.3 * np.sin(2 * np.pi * 1000 * np.linspace(0, DURATION, n))).astype(np.float32)
bird_b = (0.3 * np.sin(2 * np.pi * 3000 * np.linspace(0, DURATION, n))).astype(np.float32)

print("=== overlay ===")
print(f"bird_a  | std={bird_a.std():.4f}")
print(f"bird_b  | std={bird_b.std():.4f}")
for snr in [5, 10]:
    mixed = overlay(bird_a, bird_b, snr_db=snr)
    print(f"overlay SNR={snr}dB | std={mixed.std():.4f}  max={np.abs(mixed).max():.4f}  clipped={np.sum(np.abs(mixed) >= 1.0)}")

print("\n=== mixup ===")
label_a = np.array([1.0, 0.0])
label_b = np.array([0.0, 1.0])
for alpha in [0.2, 0.5]:
    mixed_audio, mixed_label = mixup(bird_a, label_a, bird_b, label_b, alpha=alpha)
    print(f"alpha={alpha} | label={mixed_label}  audio_std={mixed_audio.std():.4f}  (labels sum to {mixed_label.sum():.2f})")
