"""Sanity check: load an audio file and inspect chunks."""

import numpy as np
from src.data_io.audio import load_audio, chunk_audio, SAMPLE_RATE, CHUNK_DURATION

FILE = "data/birdclef-2026/train_audio/22956/XC900618.ogg"

audio = load_audio(FILE)
print(f"Loaded audio")
print(f"  shape:    {audio.shape}")
print(f"  dtype:    {audio.dtype}")
print(f"  duration: {len(audio) / SAMPLE_RATE:.2f}s")
print(f"  min/max:  {audio.min():.4f} / {audio.max():.4f}")

chunks = chunk_audio(audio, sr=SAMPLE_RATE, chunk_duration=CHUNK_DURATION)
print(f"\nChunked into {len(chunks)} x {CHUNK_DURATION}s windows")
print(f"  chunk shape: {chunks[0].shape}  (= {SAMPLE_RATE} * {CHUNK_DURATION})")
print(f"  last chunk padded: {not np.any(chunks[-1][int(len(audio) % (SAMPLE_RATE * CHUNK_DURATION)):])} "
      f"(tail zeros if audio doesn't divide evenly)")
