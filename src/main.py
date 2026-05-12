from src.models.birdnet import BirdNetEmbedder

FILE = "data/birdclef-2026/train_audio/22956/XC900618.ogg"

if __name__ == "__main__":
    embedder = BirdNetEmbedder()
    embeddings = embedder.embed_file(FILE)

    print(f"File:          {FILE}")
    print(f"Chunks:        {len(embeddings)}, each {embeddings[0].shape}")
    print(f"First vector:  {embeddings[0][0]}")
