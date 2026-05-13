from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    model_version: str = "2.4"
    backend: str = "pb"
    batch_size: int = 64
    labelled_dir: str = "data/birdclef-2026/train_audio"
    soundscape_dir: str = "data/birdclef-2026/train_soundscapes"
    output_dir: str = "data/embeddings"
