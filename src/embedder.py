from sentence_transformers import SentenceTransformer
from typing import List

# This model is free, runs locally, no API key needed
# Downloads ~90MB on first run, cached after that
MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of text strings into a list of vectors."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Embed a single string — used for query embedding."""
        return self.model.encode([text])[0].tolist()


if __name__ == "__main__":
    embedder = Embedder()
    vec = embedder.embed_single("What is the sick leave policy?")
    print(f"Embedding dimension: {len(vec)}")
    print(f"First 10 values: {vec[:10]}")