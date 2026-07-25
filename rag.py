"""
rag.py
------
Shared library for the RAG pipeline.

All the numbered pipeline scripts (01_documents.py ... 07_prompting.py) and
streamlit_app.py import functions/constants from this file. Keeping the
logic here avoids duplicating code and makes streamlit_app.py simple.

IMPORTANT: No real API key is ever hard-coded here.
OPENROUTER_API_KEY is read from an environment variable by default and is
overridden from Streamlit secrets at deploy time (see streamlit_app.py).
"""

import os
import re
import json
import glob

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "rag_collection"

CHUNK_SIZE = 500      # characters per chunk
CHUNK_OVERLAP = 80    # character overlap between consecutive chunks
TOP_K = 4              # number of chunks retrieved per query

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# API key / model config
# Never write a real key here. It stays empty until set by env var or by
# Streamlit secrets (st.secrets) in streamlit_app.py.
# --------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


# --------------------------------------------------------------------------
# Stage 1: Document loading
# --------------------------------------------------------------------------
def load_documents(folder=RAW_DOCS_DIR):
    """
    Load every .txt file in `folder` into a list of document dicts:
    {"id": "...", "source": "filename.txt", "text": "..."}
    """
    os.makedirs(folder, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    documents = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        documents.append({
            "id": f"doc_{i}",
            "source": os.path.basename(path),
            "text": text,
        })
    return documents


# --------------------------------------------------------------------------
# Stage 2: Preprocessing
# --------------------------------------------------------------------------
def preprocess_text(text):
    """
    Basic cleanup: normalize whitespace, strip control characters,
    collapse multiple blank lines.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)          # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse blank lines
    text = text.strip()
    return text


def preprocess_documents(documents):
    for doc in documents:
        doc["text"] = preprocess_text(doc["text"])
    return documents


# --------------------------------------------------------------------------
# Stage 3: Chunking
# --------------------------------------------------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple sliding-window character chunker with overlap.
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward, keeping some overlap
    return chunks


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Returns a flat list of chunk dicts:
    {"chunk_id": "...", "source": "...", "text": "..."}
    """
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for j, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{doc['id']}_chunk_{j}",
                "source": doc["source"],
                "text": piece,
            })
    return all_chunks


# --------------------------------------------------------------------------
# Stage 4: Vector representation (embeddings)
# --------------------------------------------------------------------------
_embedding_model = None


def get_embedding_model():
    """
    Lazily load and cache the sentence-transformers embedding model.
    Using a local embedding model means no extra paid API key is needed
    just to turn text into vectors.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def embed_texts(texts):
    """
    Embed a list of strings, returns a list of float lists (JSON-friendly).
    """
    model = get_embedding_model()
    vectors = model.encode(list(texts), show_progress_bar=False)
    return [v.tolist() for v in vectors]


# --------------------------------------------------------------------------
# Stage 5: Chroma vector store
# --------------------------------------------------------------------------
def get_chroma_collection(persist_directory=CHROMA_DIR, name=COLLECTION_NAME):
    import chromadb
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=name)
    return collection


def add_chunks_to_store(chunks, embeddings, collection=None):
    """
    chunks: list of {"chunk_id", "source", "text"}
    embeddings: list of float lists, same length/order as chunks
    """
    if collection is None:
        collection = get_chroma_collection()

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


# --------------------------------------------------------------------------
# Stage 6: Context retrieval
# --------------------------------------------------------------------------
def retrieve_context(query, k=TOP_K, collection=None):
    """
    Embed the query, search Chroma, and return a list of:
    {"text": ..., "source": ..., "distance": ...}
    """
    if collection is None:
        collection = get_chroma_collection()

    query_vector = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
    )

    contexts = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, dists):
        contexts.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "distance": dist,
        })
    return contexts


# --------------------------------------------------------------------------
# Stage 7: Prompting + generation
# --------------------------------------------------------------------------
def build_prompt(query, contexts):
    """
    Build a grounded prompt instructing the model to answer only from
    the retrieved context and to cite sources.
    """
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in contexts
    )

    prompt = f"""You are a helpful assistant answering questions using ONLY the context provided below.
If the answer is not contained in the context, say you don't know.
Always cite the source filename(s) you used in your answer, like (Source: filename.txt).

Context:
{context_block}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query, contexts, api_key=None, model=None):
    """
    Calls OpenRouter's chat completions endpoint with the grounded prompt
    and returns the model's answer text.
    """
    import requests

    api_key = api_key or OPENROUTER_API_KEY
    model = model or OPENROUTER_MODEL

    if not api_key:
        return ("[No API key configured. Set OPENROUTER_API_KEY in your "
                "environment or Streamlit secrets to get a real answer.]")

    prompt = build_prompt(query, contexts)

    response = requests.post(
        OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------
# Small helpers for saving/loading intermediate JSON between pipeline stages
# --------------------------------------------------------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
