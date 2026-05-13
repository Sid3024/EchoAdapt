from dataclasses import dataclass


@dataclass
class RunConfig:
    labelled_dir: str
    soundscape_dir: str
    output_dir: str
    n_trials: int
    n_audio_files: int
    subsample_n: int
    stability_lambda: float = 0.3
    seed: int = 42
    n_initial_trials: int = 10

    @classmethod
    def test(cls) -> "RunConfig":
        return cls(
            labelled_dir="data/birdclef-2026/train_audio/22930",
            soundscape_dir="data/birdclef-2026/train_audio/22985",
            output_dir="data/test/embeddings",
            n_trials=10,
            n_initial_trials=3,
            n_audio_files=5,
            subsample_n=20,
        )

    @classmethod
    def full(cls) -> "RunConfig":
        return cls(
            labelled_dir="data/birdclef-2026/train_audio",
            soundscape_dir="data/birdclef-2026/train_soundscapes",
            output_dir="data/embeddings",
            n_trials=50,
            n_audio_files=4000,
            subsample_n=50_000,
        )
