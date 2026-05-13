from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    chunk_duration: float = 5.0
