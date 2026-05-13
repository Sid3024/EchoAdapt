from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.models.birdnet import BirdNetEmbedder

CACHE_DIR = Path("data/embeddings")


def embed_folder(
    folder: str,
    embedder: BirdNetEmbedder,
    batch_size: int = 64,
    cache_every_n: int = 10,
) -> tuple[np.ndarray, list[dict]]:
    """Embed every .ogg file found recursively under folder.

    Audio chunks are accumulated until `batch_size` is reached, then sent to
    BirdNET in one encode_arrays call (one worker-process startup per batch).
    A checkpoint is written every `cache_every_n` batches.

    Works for both folder layouts:
      - train_audio/{species_id}/*.ogg  (nested by species)
      - train_soundscapes/*.ogg         (flat)

    Returns:
        embeddings  np.ndarray of shape (total_segments, 1024)
        meta        list[dict] with keys: filepath, chunk_idx, seg_idx
    """
    paths = sorted(Path(folder).rglob("*.ogg"))
    if not paths:
        raise FileNotFoundError(f"No .ogg files found under {folder!r}")

    name = Path(folder).name
    ckpt_dir = CACHE_DIR / f"{name}_partial"
    ckpt_state_path = ckpt_dir / "state.json"

    resume_file_idx = 0
    resume_chunk_idx = 0
    n_saved = 0
    ckpt_count = 0
    all_embeddings: list[np.ndarray] = []
    all_meta: list[dict] = []

    if ckpt_state_path.exists():
        state = json.loads(ckpt_state_path.read_text())
        resume_file_idx = state["file_idx"]
        resume_chunk_idx = state["chunk_idx"]

        if "n_ckpts" in state:
            ckpt_count = state["n_ckpts"]
            n_saved = state["n_saved"]
            parts: list[np.ndarray] = []
            for i in range(ckpt_count):
                parts.append(np.load(ckpt_dir / f"emb_{i:04d}.npy"))
                all_meta.extend(json.loads((ckpt_dir / f"meta_{i:04d}.json").read_text()))
            all_embeddings = list(np.concatenate(parts)) if parts else []
        else:
            # Migrate old single-file format to numbered format
            old_emb = ckpt_dir / "embeddings.npy"
            old_meta = ckpt_dir / "meta.json"
            if old_emb.exists():
                all_embeddings = list(np.load(old_emb))
                all_meta = json.loads(old_meta.read_text())
                np.save(ckpt_dir / "emb_0000.npy", np.stack(all_embeddings))
                (ckpt_dir / "meta_0000.json").write_text(json.dumps(all_meta))
                old_emb.unlink()
                old_meta.unlink()
            ckpt_count = 1 if all_embeddings else 0
            n_saved = len(all_embeddings)
            ckpt_state_path.write_text(json.dumps({
                "file_idx": resume_file_idx,
                "chunk_idx": resume_chunk_idx,
                "n_ckpts": ckpt_count,
                "n_saved": n_saved,
            }))

        print(
            f"Resuming from file {resume_file_idx}/{len(paths)}, chunk {resume_chunk_idx} "
            f"({len(all_embeddings):,} segments across {ckpt_count} checkpoints)"
        )

    batch: list[tuple[np.ndarray, int, int]] = []  # (audio_chunk, file_idx, chunk_idx)
    last_file_idx = resume_file_idx
    last_chunk_idx = resume_chunk_idx - 1
    batches_since_ckpt = 0

    with embedder.open_session(batch_size=batch_size) as session:
        for file_idx, path in enumerate(paths):
            if file_idx < resume_file_idx:
                continue

            print(f"[{file_idx + 1}/{len(paths)}] {path}", flush=True)
            chunks = embedder.get_chunks(str(path))
            start_chunk = resume_chunk_idx if file_idx == resume_file_idx else 0

            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx < start_chunk:
                    continue
                batch.append((chunk, file_idx, chunk_idx))

                if len(batch) == batch_size:
                    _process_batch(batch, embedder, batch_size, all_embeddings, all_meta, paths, session)
                    last_file_idx = batch[-1][1]
                    last_chunk_idx = batch[-1][2]
                    batches_since_ckpt += 1
                    batch = []

                    if batches_since_ckpt >= cache_every_n:
                        _save_checkpoint(
                            ckpt_dir, ckpt_state_path,
                            n_saved, ckpt_count,
                            last_file_idx, last_chunk_idx + 1,
                            all_embeddings, all_meta,
                        )
                        n_saved = len(all_embeddings)
                        ckpt_count += 1
                        batches_since_ckpt = 0

        if batch:
            _process_batch(batch, embedder, batch_size, all_embeddings, all_meta, paths, session)

    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    return np.stack(all_embeddings), all_meta


def _process_batch(
    batch: list[tuple[np.ndarray, int, int]],
    embedder: BirdNetEmbedder,
    batch_size: int,
    all_embeddings: list[np.ndarray],
    all_meta: list[dict],
    paths: list[Path],
    session=None,
) -> None:
    chunk_arrays = [item[0] for item in batch]
    print(f"  encoding {len(chunk_arrays)} chunks...", flush=True)
    if session is not None:
        encoded = embedder.encode_chunks_with_session(session, chunk_arrays)
    else:
        encoded = embedder.encode_chunks(chunk_arrays, batch_size=batch_size)
    for i, ((_, file_idx, chunk_idx), segs) in enumerate(zip(batch, encoded)):
        for seg_idx in range(len(segs)):
            all_embeddings.append(segs[seg_idx])
            all_meta.append({
                "filepath": str(paths[file_idx]),
                "chunk_idx": chunk_idx,
                "seg_idx": seg_idx,
            })
        print(f"  [{i + 1}/{len(batch)}] encoded: {paths[file_idx].name} chunk {chunk_idx} -> {len(segs)} segs", flush=True)
    print(f"  batch done: {len(all_embeddings):,} total segments", flush=True)


def _save_checkpoint(
    ckpt_dir: Path,
    state_path: Path,
    n_saved: int,
    ckpt_count: int,
    file_idx: int,
    chunk_idx: int,
    all_embeddings: list[np.ndarray],
    all_meta: list[dict],
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    new_embs = all_embeddings[n_saved:]
    new_meta = all_meta[n_saved:]
    np.save(ckpt_dir / f"emb_{ckpt_count:04d}.npy", np.stack(new_embs))
    (ckpt_dir / f"meta_{ckpt_count:04d}.json").write_text(json.dumps(new_meta))
    state_path.write_text(json.dumps({
        "file_idx": file_idx,
        "chunk_idx": chunk_idx,
        "n_ckpts": ckpt_count + 1,
        "n_saved": len(all_embeddings),
    }))
    print(
        f"  >>> checkpoint {ckpt_count} saved: +{len(new_embs):,} segments "
        f"({len(all_embeddings):,} total, next: file {file_idx} chunk {chunk_idx}) <<<",
        flush=True,
    )
