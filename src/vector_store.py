import chromadb
from typing import List

class VectorStore:
    def __init__(self, collection_name: str = "docqa"):
        # Persistent storage — saves to disk, survives restarts
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # use cosine similarity
        )
        print(f"Vector store ready. Collection: {collection_name}")

    def add_chunks(self, chunks: List[str], embeddings: List[List[float]]):
        """Store text chunks and their embeddings."""
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )
        print(f"Stored {len(chunks)} chunks in vector store.")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Find top_k most similar chunks to the query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Return just the text chunks
        return results["documents"][0]

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Clear the collection — useful when loading a new document."""
        self.client.delete_collection("docqa")
        self.collection = self.client.get_or_create_collection(
            name="docqa",
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared.")