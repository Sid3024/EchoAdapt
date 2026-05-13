"""Sanity check: build_pipeline constructs a valid pipeline and applies without error."""

import numpy as np
import optuna

from src.augment.pipeline import Compose
from src.config.augment import AugmentSearchConfig
from src.search.optimize import build_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

SR = 48000
DURATION = 3.0
t = np.linspace(0, DURATION, int(SR * DURATION), dtype=np.float32)
audio = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

aug_config = AugmentSearchConfig()

study = optuna.create_study(direction="minimize")
trial = study.ask()

pipeline, p_overlay, snr_overlay = build_pipeline(trial, aug_config)

print("=== build_pipeline ===")
print(f"pipeline type : {type(pipeline).__name__}  (expect Compose)")
print(f"n_transforms  : {len(pipeline.transforms)}  (expect 5)")
print(f"p_overlay     : {p_overlay:.4f}  (in [{aug_config.overlay.p_min}, {aug_config.overlay.p_max}])")
print(f"snr_overlay   : {snr_overlay:.4f}  (in [{aug_config.overlay.snr_min}, {aug_config.overlay.snr_max}])")

print("\n=== pipeline apply ===")
np.random.seed(0)
out = pipeline(audio)
print(f"input  len={len(audio)}  std={audio.std():.4f}")
print(f"output len={len(out)}   std={out.std():.4f}  (len preserved: {len(out) == len(audio)})")

print("\n=== multiple applications (stochastic) ===")
np.random.seed(42)
stds = [pipeline(audio).std() for _ in range(10)]
print(f"std range across 10 runs: [{min(stds):.4f}, {max(stds):.4f}]")
