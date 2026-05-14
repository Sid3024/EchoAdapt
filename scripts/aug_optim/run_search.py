"""Entry point for the augmentation search pipeline.

Smoke test (uses data/test/labelled and data/test/soundscape):
  python -m scripts.aug_optim.run_search --preset test --step all

Full run:
  python -m scripts.aug_optim.run_search --preset full --step all

Run individual steps:
  --step embed    embed both labelled and soundscape dirs
  --step prepare  subsample soundscape embeddings + cache labelled audio chunks
  --step search   run Bayesian optimisation
  --step all      embed → prepare → search in sequence

Edit RunConfig.test() / RunConfig.full() in src/config/run.py to change parameters.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# must be set before TF/BirdNET is imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

from src.config import AugmentSearchConfig, EmbeddingConfig, SearchConfig
from src.config.run import RunConfig
from src.search.embed import (
    embed_soundscapes,
    load_search_data,
    prepare_search_data,
)
from src.search.optimize import run_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmentation Bayesian search")
    p.add_argument("--preset", choices=["test", "full"], default="full",
                   help="Configuration preset (default: full)")
    p.add_argument("--step", choices=["embed", "prepare", "search", "all"],
                   required=True)
    return p.parse_args()


def build_configs(run: RunConfig) -> tuple[EmbeddingConfig, AugmentSearchConfig, SearchConfig]:
    emb = EmbeddingConfig(
        labelled_dir=run.labelled_dir,
        soundscape_dir=run.soundscape_dir,
        output_dir=run.output_dir,
    )
    search = SearchConfig(
        n_trials=run.n_trials,
        n_initial_trials=run.n_initial_trials,
        n_audio_files=run.n_audio_files,
        file_batch_size=run.file_batch_size,
        subsample_n=run.subsample_n,
        stability_lambda=run.stability_lambda,
        seed=run.seed,
    )
    return emb, AugmentSearchConfig(), search


def step_embed(emb_config: EmbeddingConfig) -> None:
    from src.models.birdnet import BirdNetEmbedder
    embedder = BirdNetEmbedder(version=emb_config.model_version, backend=emb_config.backend)
    print("=== Embedding soundscapes ===")
    embed_soundscapes(emb_config, embedder)


def step_prepare(emb_config: EmbeddingConfig, search_config: SearchConfig) -> None:
    print("=== Preparing search subsample ===")
    file_paths, sc_emb = prepare_search_data(emb_config, search_config)
    print(f"  labelled files  : {len(file_paths)}")
    print(f"  soundscape embs : {sc_emb.shape}")


def step_search(
    emb_config: EmbeddingConfig,
    aug_config: AugmentSearchConfig,
    search_config: SearchConfig,
) -> None:
    from src.models.birdnet import BirdNetEmbedder
    embedder = BirdNetEmbedder(version=emb_config.model_version, backend=emb_config.backend)

    print("=== Loading search data ===")
    file_paths, soundscape_emb = load_search_data(emb_config)
    print(f"  labelled files: {len(file_paths)}  soundscape emb: {soundscape_emb.shape}")

    print(f"=== Running Bayesian optimisation ({search_config.n_trials} trials) ===")
    study = run_search(file_paths, soundscape_emb, embedder, aug_config, search_config)

    out = Path(emb_config.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out / f"search_results_{timestamp}.json"
    results = {
        "best_score": study.best_value,
        "best_params": study.best_params,
        "labelled_dir": emb_config.labelled_dir,
        "soundscape_dir": emb_config.soundscape_dir,
        "search_config": {
            "n_trials": search_config.n_trials,
            "n_audio_files": search_config.n_audio_files,
            "subsample_n": search_config.subsample_n,
            "stability_lambda": search_config.stability_lambda,
        },
        "all_trials": [
            {"number": t.number, "score": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    args = parse_args()
    run = RunConfig.test() if args.preset == "test" else RunConfig.full()
    emb_config, aug_config, search_config = build_configs(run)

    if args.step in ("embed", "all"):
        step_embed(emb_config)
    if args.step in ("prepare", "all"):
        step_prepare(emb_config, search_config)
    if args.step in ("search", "all"):
        step_search(emb_config, aug_config, search_config)
