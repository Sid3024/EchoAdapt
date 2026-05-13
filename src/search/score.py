"""Step 2 of the search pipeline: compute the augmentation quality score.

Score = coverage + lambda * stability

  coverage:  mean distance from each soundscape embedding to its nearest
             labelled embedding. Measures how well augmented labelled data
             covers the soundscape distribution. Lower is better.

  stability: mean distance from each labelled embedding to its nearest
             soundscape embedding. Penalises augmented data drifting far
             outside the soundscape distribution.

  lambda:    hand-tunable weight in [0, 1]. Coverage always dominates.
"""

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def one_sided_chamfer(source: np.ndarray, target: np.ndarray) -> float:
    """Mean distance from each source point to its nearest target point.

    Args:
        source: (N, D)
        target: (M, D)

    Returns:
        Scalar mean nearest-neighbour distance.
    """
    _, dists = pairwise_distances_argmin_min(source, target, metric="euclidean")
    return float(np.mean(dists))


def chamfer_score(
    labelled_emb: np.ndarray,
    soundscape_emb: np.ndarray,
    stability_lambda: float = 0.3,
) -> float:
    """Weighted asymmetric Chamfer score. Lower is better.

    Args:
        labelled_emb:     Augmented labelled embeddings (N, 1024).
        soundscape_emb:   Fixed soundscape embeddings (M, 1024).
        stability_lambda: Weight for the labelled→soundscape direction.

    Returns:
        score = coverage + stability_lambda * stability
    """
    coverage = one_sided_chamfer(soundscape_emb, labelled_emb)
    stability = one_sided_chamfer(labelled_emb, soundscape_emb)
    return coverage + stability_lambda * stability
