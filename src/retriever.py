from src.embedder import Embedder
from src.vector_store import VectorStore
from typing import List


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, top_k: int = 5) -> List[str]:
        """
        Embed the question and retrieve top_k
        most relevant chunks from the vector store.
        """
        print(f"🔍 Retrieving top {top_k} chunks for question...")
        query_embedding = self.embedder.embed_single(question)
        chunks = self.store.query(query_embedding, top_k=top_k)
        return chunks

    def retrieve_with_scores(self, question: str, top_k: int = 5) -> List[dict]:
        """
        Same as retrieve() but also returns similarity scores.
        Useful for debugging — lets you see HOW relevant each chunk is.
        """
        query_embedding = self.embedder.embed_single(question)

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"]
        )

        chunks = results["documents"][0]
        # ChromaDB returns distances, not similarities
        # For cosine: similarity = 1 - distance
        distances = results["distances"][0]

        return [
            {
                "chunk": chunk,
                "similarity": round(1 - dist, 4)
            }
            for chunk, dist in zip(chunks, distances)
        ]


if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("Vector store is empty. Run main.py first to index a document.")
    else:
        retriever = Retriever(embedder, store)
        results = retriever.retrieve_with_scores("What is the sick leave policy?")
        for i, r in enumerate(results):
            print(f"\nChunk {i+1} | Similarity: {r['similarity']}")
            print(r["chunk"][:200], "...")