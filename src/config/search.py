from dataclasses import dataclass


@dataclass
class SearchConfig:
    n_initial_trials: int = 10     # random trials before GP kicks in
    n_trials: int = 50             # total budget including initial
    stability_lambda: float = 0.3  # weight for labelled→soundscape direction
    subsample_n: int = 1000        # soundscape embeddings used per score computation
    n_audio_files: int = 200       # labelled files selected for augmentation
    file_batch_size: int = 50      # files loaded per batch during each trial
    seed: int = 42
