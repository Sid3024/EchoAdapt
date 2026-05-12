import numpy as np
import umap
import matplotlib.pyplot as plt


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Fit UMAP on embeddings and return 2D coordinates."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
    )
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords: np.ndarray,
    labels: list[str],
    title: str = "UMAP",
    save_path: str | None = None,
) -> None:
    """Scatter plot of 2D UMAP coords coloured by label."""
    unique = sorted(set(labels))
    label_arr = np.array(labels)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(unique):
        mask = label_arr == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i % len(colors)]],
            label=f"{label} (n={mask.sum():,})",
            alpha=0.4,
            s=4,
            linewidths=0,
        )

    ax.legend(markerscale=4, fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", "datalim")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()
