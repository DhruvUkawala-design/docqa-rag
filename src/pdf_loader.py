import PyPDF2
import os

def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    text=""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    overlap ensures context isn't lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap by 50 words

    return chunks

if __name__ == "__main__":
    # Quick test
    sample_text = "word " * 1200
    chunks = chunk_text(sample_text)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk word count: {len(chunks[0].split())}")
    print(f"Second chunk starts at word: {chunks[1].split()[0]}")