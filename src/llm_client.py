import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

def check_ollama_running() -> bool:
    """Check if Ollama service is up before making requests."""
    try:
        response = requests.get("http://localhost:11434", timeout=500)
        return True
    except:
        return False

def ask_llm(question: str, context_chunks: list[str]) -> str:
    """
    Send question + retrieved context to local LLM.
    Returns the generated answer.
    """
    # Check Ollama is running first
    if not check_ollama_running():
        return "ERROR: Ollama is not running. Open a new terminal and run: ollama serve"

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I could not find this information in the document."
Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    print("⏳ Waiting for LLM response (may take 30-60 seconds on first run)...")

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=1500)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Open a new terminal and run: ollama serve"
    except requests.exceptions.Timeout:
        return "ERROR: LLM timed out. Try again — first response is always slowest."
    except Exception as e:
        return f"ERROR: {str(e)}"