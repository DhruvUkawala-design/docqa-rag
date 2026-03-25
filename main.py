import os
import sys
from src.pdf_loader import load_pdf, chunk_text
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.llm_client import ask_llm

def index_document(pdf_path: str):
    """Load PDF, chunk it, embed chunks, store in vector DB."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    text = load_pdf(pdf_path)
    print(f"Extracted {len(text.split())} words from PDF.")

    print("\n✂️  Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks.")

    print("\n🔢 Embedding chunks...")
    embedder = Embedder()
    embeddings = embedder.embed(chunks)

    print("\n💾 Storing in vector database...")
    store = VectorStore()
    store.reset()  # clear old data
    store.add_chunks(chunks, embeddings)
    print(f"✅ Indexed {store.count()} chunks. Ready to query.\n")

    return embedder, store


def query_document(question: str, embedder: Embedder, store: VectorStore):
    """Embed question, retrieve relevant chunks, generate answer."""
    print(f"\n❓ Question: {question}")

    retriever = Retriever(embedder, store)
    top_chunks = retriever.retrieve(question, top_k=5)

    print(f"📚 Retrieved {len(top_chunks)} relevant chunks.")
    print("\n🤖 Asking LLM...")
    answer = ask_llm(question, top_chunks)

    print(f"\n💡 Answer:\n{answer}")
    return answer


def main():
    print("=" * 50)
    print("       DocQA — RAG Document Q&A System")
    print("=" * 50)

    # Step 1: Get PDF path
    pdf_path = input("\nEnter path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    # Step 2: Index the document
    embedder, store = index_document(pdf_path)

    # Step 3: Question loop
    print("Ask questions about your document. Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if question:
            query_document(question, embedder, store)
            print("\n" + "-" * 50)


if __name__ == "__main__":
    main()

