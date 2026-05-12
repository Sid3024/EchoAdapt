from src.models.birdnet import BirdNetEmbedder

FILE = "data/birdclef-2026/train_audio/22956/XC900618.ogg"

if __name__ == "__main__":
    embedder = BirdNetEmbedder()
    embeddings = embedder.embed_file(FILE)

    print(f"File:            {FILE}")
    print(f"Embedding shape: {embeddings.shape}")  # (n_chunks, 1024)
    print(f"First vector:    {embeddings[0]}")
