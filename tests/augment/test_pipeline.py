"""Sanity check: Compose and RandomApply behaviour."""

import numpy as np
from src.augment.pipeline import Compose, RandomApply

SR = 48000
DURATION = 3.0
t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
audio = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

print("=== RandomApply ===")
always = RandomApply(lambda a: a * 2.0, p=1.0)
never = RandomApply(lambda a: a * 2.0, p=0.0)
print(f"p=1.0 | std ratio (expect 2.0): {always(audio).std() / audio.std():.4f}")
print(f"p=0.0 | unchanged (expect True): {np.array_equal(never(audio), audio)}")

print("\n=== Compose ===")
double = lambda a: a * 2.0  # noqa: E731
halve = lambda a: a * 0.5   # noqa: E731
pipeline = Compose([double, halve])
out = pipeline(audio)
print(f"double→halve | unchanged (expect True): {np.allclose(out, audio)}")

pipeline_two_doubles = Compose([double, double])
out2 = pipeline_two_doubles(audio)
print(f"double→double | std ratio (expect 4.0): {out2.std() / audio.std():.4f}")

print("\n=== RandomApply inside Compose ===")
np.random.seed(0)
sometimes = RandomApply(lambda a: a * 10.0, p=0.5)
results = [sometimes(audio).std() / audio.std() for _ in range(1000)]
applied = sum(1 for r in results if r > 5.0)
print(f"p=0.5 applied ~500/1000 times: {applied}  (expect ~500)")
