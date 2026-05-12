from src.models.birdnet import BirdNetEmbedder
from src.data_io.folder import embed_folder
if __name__ == "__main__":
    embedder = BirdNetEmbedder()
    emb, meta = embed_folder("data/birdclef-2026/train_audio/23150", embedder)

    print(f"Embeddings shape: {emb.shape}")   # (total_segments, 1024)
    print(f"Meta entries:     {len(meta)}")
    print(f"First entry:      {meta[0]}")

#from src.models.birdnet import BirdNetEmbedder
# FILE = "data/birdclef-2026/train_audio/22956/XC900618.ogg"

# if __name__ == "__main__":
#     embedder = BirdNetEmbedder()
#     embeddings = embedder.embed_file(FILE)

#     print(f"File:          {FILE}")
#     print(f"Chunks:        {len(embeddings)}, each {embeddings[0].shape}")
#     print(f"First vector:  {embeddings[0][0]}")
