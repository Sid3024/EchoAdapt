"""Entry point for the augmentation search pipeline.

Run steps in order (each step saves to disk so you can resume):

  Step 1 — embed labelled + soundscape data (requires GPU):
    python -m src.search.run --step embed

  Step 2 — prepare fixed search subsample from saved embeddings:
    python -m src.search.run --step prepare

  Step 3 — run Bayesian optimisation (requires GPU for BirdNET):
    python -m src.search.run --step search
"""

import argparse
import json
from pathlib import Path

from src.config import EmbeddingConfig, AugmentSearchConfig, SearchConfig
from src.search.embed import embed_labelled, embed_soundscapes, prepare_search_data, load_search_data
from src.search.optimize import run_search


def get_configs():
    return EmbeddingConfig(), AugmentSearchConfig(), SearchConfig()


def step_embed(emb_config: EmbeddingConfig) -> None:
    from src.models.birdnet import BirdNetEmbedder
    embedder = BirdNetEmbedder(version=emb_config.model_version, backend=emb_config.backend)
    print("=== Embedding labelled data ===")
    embed_labelled(emb_config, embedder)
    print("=== Embedding soundscapes ===")
    embed_soundscapes(emb_config, embedder)


def step_prepare(emb_config: EmbeddingConfig, search_config: SearchConfig) -> None:
    print("=== Preparing search subsample ===")
    prepare_search_data(emb_config, search_config)


def step_search(
    emb_config: EmbeddingConfig,
    aug_config: AugmentSearchConfig,
    search_config: SearchConfig,
) -> None:
    from src.models.birdnet import BirdNetEmbedder
    embedder = BirdNetEmbedder(version=emb_config.model_version, backend=emb_config.backend)

    print("=== Loading search data ===")
    labelled_chunks, soundscape_emb = load_search_data(emb_config)
    print(f"  labelled chunks: {labelled_chunks.shape}, soundscape emb: {soundscape_emb.shape}")

    print("=== Running Bayesian optimisation ===")
    study = run_search(labelled_chunks, soundscape_emb, embedder, aug_config, search_config)

    out = Path(emb_config.output_dir)
    results_path = out / "search_results.json"
    results = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "score": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["embed", "prepare", "search"], required=True)
    args = parser.parse_args()

    emb_config, aug_config, search_config = get_configs()

    if args.step == "embed":
        step_embed(emb_config)
    elif args.step == "prepare":
        step_prepare(emb_config, search_config)
    elif args.step == "search":
        step_search(emb_config, aug_config, search_config)
