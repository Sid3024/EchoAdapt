from dataclasses import dataclass, field


@dataclass
class PinkNoiseConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    snr_min: float = 5.0
    snr_max: float = 20.0


@dataclass
class GaussianNoiseConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    snr_min: float = 5.0
    snr_max: float = 20.0


@dataclass
class GainConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    min_db_min: float = -12.0   # lower bound for the gain floor
    min_db_max: float = 0.0     # upper bound for the gain floor
    max_db_min: float = 0.0     # lower bound for the gain ceiling
    max_db_max: float = 12.0    # upper bound for the gain ceiling


@dataclass
class PitchShiftConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    steps_min: float = -3.0     # max downward shift (semitones)
    steps_max: float = 3.0      # max upward shift (semitones)


@dataclass
class TimeStretchConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    rate_min: float = 0.8       # max slowdown
    rate_max: float = 1.2       # max speedup


@dataclass
class OverlayConfig:
    p_min: float = 0.0
    p_max: float = 1.0
    snr_min: float = 3.0
    snr_max: float = 15.0


@dataclass
class AugmentSearchConfig:
    pink_noise: PinkNoiseConfig = field(default_factory=PinkNoiseConfig)
    gaussian_noise: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    pitch_shift: PitchShiftConfig = field(default_factory=PitchShiftConfig)
    time_stretch: TimeStretchConfig = field(default_factory=TimeStretchConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
