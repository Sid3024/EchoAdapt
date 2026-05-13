"""Step 3 of the search pipeline: Bayesian optimisation over augmentation configs.

Optuna's TPE sampler acts as the GP surrogate — it models the score surface
and picks the next config to evaluate using expected improvement.
"""

from __future__ import annotations

import numpy as np
import optuna

from src.augment import (
    Compose, RandomApply,
    add_pink_noise, add_gaussian_noise,
    random_gain, pitch_shift, time_stretch,
    overlay,
)
from src.config.augment import AugmentSearchConfig
from src.config.search import SearchConfig
from src.search.score import chamfer_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_pipeline(trial: optuna.Trial, aug_config: AugmentSearchConfig) -> Compose:
    """Construct an augmentation pipeline from a single Optuna trial's parameters."""
    c = aug_config
    transforms = []

    # Pink noise
    p_pink = trial.suggest_float("p_pink", c.pink_noise.p_min, c.pink_noise.p_max)
    snr_pink = trial.suggest_float("snr_pink", c.pink_noise.snr_min, c.pink_noise.snr_max)
    transforms.append(RandomApply(lambda a, s=snr_pink: add_pink_noise(a, s), p=p_pink))

    # Gaussian noise
    p_gauss = trial.suggest_float("p_gauss", c.gaussian_noise.p_min, c.gaussian_noise.p_max)
    snr_gauss = trial.suggest_float("snr_gauss", c.gaussian_noise.snr_min, c.gaussian_noise.snr_max)
    transforms.append(RandomApply(lambda a, s=snr_gauss: add_gaussian_noise(a, s), p=p_gauss))

    # Gain
    p_gain = trial.suggest_float("p_gain", c.gain.p_min, c.gain.p_max)
    gain_min = trial.suggest_float("gain_min_db", c.gain.min_db_min, c.gain.min_db_max)
    gain_max = trial.suggest_float("gain_max_db", c.gain.max_db_min, c.gain.max_db_max)
    transforms.append(RandomApply(lambda a, mn=gain_min, mx=gain_max: random_gain(a, mn, mx), p=p_gain))

    # Pitch shift
    p_pitch = trial.suggest_float("p_pitch", c.pitch_shift.p_min, c.pitch_shift.p_max)
    pitch_min = trial.suggest_float("pitch_min", c.pitch_shift.steps_min, 0.0)
    pitch_max = trial.suggest_float("pitch_max", 0.0, c.pitch_shift.steps_max)
    transforms.append(
        RandomApply(lambda a, mn=pitch_min, mx=pitch_max: pitch_shift(a, n_steps=float(np.random.uniform(mn, mx))), p=p_pitch)
    )

    # Time stretch
    p_stretch = trial.suggest_float("p_stretch", c.time_stretch.p_min, c.time_stretch.p_max)
    stretch_min = trial.suggest_float("stretch_min", c.time_stretch.rate_min, 1.0)
    stretch_max = trial.suggest_float("stretch_max", 1.0, c.time_stretch.rate_max)
    transforms.append(
        RandomApply(lambda a, mn=stretch_min, mx=stretch_max: time_stretch(a, rate=float(np.random.uniform(mn, mx))), p=p_stretch)
    )

    # Overlay (second source sampled randomly from the labelled pool at call time)
    p_overlay = trial.suggest_float("p_overlay", c.overlay.p_min, c.overlay.p_max)
    snr_overlay = trial.suggest_float("snr_overlay", c.overlay.snr_min, c.overlay.snr_max)
    transforms.append(
        RandomApply(lambda a, s=snr_overlay: a, p=p_overlay)  # placeholder; resolved in evaluate_trial
    )

    return Compose(transforms[:-1]), p_overlay, snr_overlay  # overlay handled separately


def evaluate_trial(
    trial: optuna.Trial,
    labelled_chunks: np.ndarray,
    soundscape_emb: np.ndarray,
    embedder,
    aug_config: AugmentSearchConfig,
    search_config: SearchConfig,
) -> float:
    """Augment labelled chunks, embed, compute chamfer score."""
    base_pipeline, p_overlay, snr_overlay = build_pipeline(trial, aug_config)

    augmented = []
    for i, chunk in enumerate(labelled_chunks):
        aug = base_pipeline(chunk)

        # overlay: mix with a randomly chosen other chunk from the pool
        if np.random.random() < p_overlay:
            other_idx = np.random.randint(0, len(labelled_chunks))
            aug = overlay(aug, labelled_chunks[other_idx], snr_db=snr_overlay)

        augmented.append(aug)

    encoded = embedder.encode_chunks(augmented)
    labelled_emb = np.vstack(encoded)  # (total_segs, 1024)

    score = chamfer_score(labelled_emb, soundscape_emb, search_config.stability_lambda)
    print(f"  trial {trial.number:>3d} | score={score:.4f}", flush=True)
    return score


def run_search(
    labelled_chunks: np.ndarray,
    soundscape_emb: np.ndarray,
    embedder,
    aug_config: AugmentSearchConfig,
    search_config: SearchConfig,
) -> optuna.Study:
    """Run Bayesian optimisation and return the completed Optuna study."""

    def objective(trial: optuna.Trial) -> float:
        return evaluate_trial(trial, labelled_chunks, soundscape_emb, embedder, aug_config, search_config)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=search_config.n_initial_trials,
        seed=search_config.seed,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=search_config.n_trials)

    print(f"\nBest score:  {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study
