"""Perceptual check: apply each augmentation to a real file and save outputs.

All outputs saved to data/aug_tests/file_1/.
Listen to each file and verify the effect matches the description in the comments.

Usage:
    python -m tests.augment.test_aug_listen
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from src.data_io.audio import load_audio, SAMPLE_RATE
from src.augment.noise import add_gaussian_noise, add_pink_noise
from src.augment.transforms import random_gain, pitch_shift, time_stretch
from src.augment.mix import overlay

OUT = Path("data/aug_tests/file_1")
SR = SAMPLE_RATE

audio = load_audio(str(OUT / "iNat317238.ogg"))
print(f"Loaded: {len(audio)/SR:.1f}s  ({len(audio)} samples)")


def save(name: str, data: np.ndarray) -> None:
    path = OUT / name
    sf.write(path, data, SR)
    print(f"  saved {name}")


# --- noise ---
save("gaussian_snr20.ogg", add_gaussian_noise(audio, snr_db=20))  # gentle hiss barely audible under the bird
save("gaussian_snr10.ogg", add_gaussian_noise(audio, snr_db=10))  # noticeable white hiss, bird still clear
save("gaussian_snr5.ogg",  add_gaussian_noise(audio, snr_db=5))   # heavy white noise, bird partially masked

save("pink_snr20.ogg", add_pink_noise(audio, snr_db=20))  # soft low-rumble ambience, bird dominant
save("pink_snr10.ogg", add_pink_noise(audio, snr_db=10))  # audible wind-like background, bird clear
save("pink_snr5.ogg",  add_pink_noise(audio, snr_db=5))   # strong low-frequency rumble, bird competing

# --- gain ---
save("gain_minus6db.ogg", random_gain(audio, min_db=-6, max_db=-6))  # noticeably quieter (~half volume)
save("gain_plus6db.ogg",  random_gain(audio, min_db=6,  max_db=6))   # noticeably louder (~double volume), may clip

# --- pitch shift ---
save("pitch_minus2st.ogg", pitch_shift(audio, sr=SR, n_steps=-2))  # bird call sounds lower / deeper
save("pitch_plus2st.ogg",  pitch_shift(audio, sr=SR, n_steps=2))   # bird call sounds higher / squeakier

# --- time stretch ---
save("stretch_slow.ogg",   time_stretch(audio, rate=0.9))  # slightly slower, calls a bit drawn out
save("stretch_fast.ogg",   time_stretch(audio, rate=1.1))  # slightly faster, calls a bit compressed

# --- overlay (bird mixed with itself offset as a stand-in for a second species) ---
np.random.seed(0)
save("overlay_snr10.ogg", overlay(audio, np.roll(audio, SR // 2), snr_db=10))  # faint second bird underneath
save("overlay_snr3.ogg",  overlay(audio, np.roll(audio, SR // 2), snr_db=3))   # second bird nearly as loud as first

print("\nDone. Files written to", OUT)
