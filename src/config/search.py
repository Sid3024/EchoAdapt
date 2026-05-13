from dataclasses import dataclass


@dataclass
class SearchConfig:
    n_initial_trials: int = 10     # random trials before GP kicks in
    n_trials: int = 50             # total budget including initial
    stability_lambda: float = 0.3  # weight for labelled→soundscape direction
    subsample_n: int = 1000        # soundscape embeddings used per score computation
    n_audio_files: int = 200       # labelled files loaded as raw audio for augmentation
    seed: int = 42
