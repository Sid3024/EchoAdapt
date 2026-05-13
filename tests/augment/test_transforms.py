"""Sanity check: apply gain, pitch shift, time stretch and inspect output."""

import numpy as np
from src.augment.transforms import random_gain, pitch_shift, time_stretch
from src.data_io.audio import SAMPLE_RATE

SR = SAMPLE_RATE
DURATION = 3.0
t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
audio = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

print(f"Original     | len={len(audio)}  std={audio.std():.4f}")

for db in [-6, 0, 6]:
    out = random_gain(audio, min_db=db, max_db=db)
    print(f"Gain {db:+d}dB    | len={len(out)}  std={out.std():.4f}  (expected ~{audio.std() * 10**(db/20):.4f})")

print()
for steps in [-2, 0, 2]:
    out = pitch_shift(audio, sr=SR, n_steps=steps)
    print(f"Pitch {steps:+d} st  | len={len(out)}  std={out.std():.4f}")

print()
for rate in [0.9, 1.0, 1.1]:
    out = time_stretch(audio, rate=rate)
    print(f"Stretch {rate:.1f}x  | len={len(out)}  (same as input: {len(out) == len(audio)})")
