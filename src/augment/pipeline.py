from collections.abc import Callable

import numpy as np

AudioTransform = Callable[[np.ndarray], np.ndarray]


class RandomApply:
    """Apply a transform with probability p, otherwise pass through unchanged."""

    def __init__(self, transform: AudioTransform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.transform(audio)
        return audio


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: list[AudioTransform]):
        self.transforms = transforms

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            audio = t(audio)
        return audio
