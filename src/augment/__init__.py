from src.augment.noise import add_gaussian_noise, add_pink_noise, add_background
from src.augment.transforms import random_gain, pitch_shift, time_stretch
from src.augment.mix import overlay, mixup
from src.augment.pipeline import RandomApply, Compose
