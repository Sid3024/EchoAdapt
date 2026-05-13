"""Sanity check: chamfer score with controlled synthetic embeddings."""

import numpy as np
from src.search.score import one_sided_chamfer, chamfer_score

rng = np.random.default_rng(0)
DIM = 1024

# Case 1: identical distributions — score should be near 0
a = rng.standard_normal((500, DIM)).astype(np.float32)
b = a + rng.standard_normal((500, DIM)).astype(np.float32) * 0.001
print(f"Identical distributions")
print(f"  coverage (sc→lab): {one_sided_chamfer(b, a):.4f}  (expect ~0)")
print(f"  stability (lab→sc): {one_sided_chamfer(a, b):.4f}  (expect ~0)")
print(f"  chamfer score:     {chamfer_score(a, b, stability_lambda=0.3):.4f}")

print()

# Case 2: labelled is a tight subset of soundscape — coverage should be high
soundscape = rng.standard_normal((500, DIM)).astype(np.float32)
labelled_subset = soundscape[:50] + rng.standard_normal((50, DIM)).astype(np.float32) * 0.001
print(f"Labelled = tight subset of soundscape (bad coverage)")
print(f"  coverage (sc→lab): {one_sided_chamfer(soundscape, labelled_subset):.4f}  (expect high)")
print(f"  stability (lab→sc): {one_sided_chamfer(labelled_subset, soundscape):.4f}  (expect low)")
print(f"  chamfer score:     {chamfer_score(labelled_subset, soundscape, stability_lambda=0.3):.4f}")

print()

# Case 3: labelled covers soundscape well — score should be low
labelled_covering = rng.standard_normal((500, DIM)).astype(np.float32) * soundscape.std()
print(f"Labelled covers soundscape well")
print(f"  coverage (sc→lab): {one_sided_chamfer(soundscape, labelled_covering):.4f}  (expect low)")
print(f"  stability (lab→sc): {one_sided_chamfer(labelled_covering, soundscape):.4f}")
print(f"  chamfer score:     {chamfer_score(labelled_covering, soundscape, stability_lambda=0.3):.4f}")
