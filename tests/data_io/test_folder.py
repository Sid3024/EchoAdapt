from src.models.birdnet import BirdNetEmbedder
from src.data_io.folder import embed_folder
if __name__ == "__main__":
    embedder = BirdNetEmbedder()
    emb, meta = embed_folder("data/birdclef-2026/train_audio/23150", embedder)

    print(f"Embeddings shape: {emb.shape}")   # (total_segments, 1024)
    print(f"Meta entries:     {len(meta)}")
    print(f"First entry:      {meta[0]}")
