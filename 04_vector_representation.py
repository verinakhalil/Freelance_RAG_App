"""
04_vector_representation.py
-----------------------------
Stage 4 of the RAG pipeline: turn text chunks into embedding vectors
using a local sentence-transformers model (no API key required).

Run after 03_chunking.py:
    python 04_vector_representation.py
"""

import rag

def main():
    in_path = f"{rag.DATA_DIR}/03_chunks.json"
    chunks = rag.load_json(in_path)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunk(s) with '{rag.EMBEDDING_MODEL_NAME}'...")

    embeddings = rag.embed_texts(texts)

    print(f"Done. Each vector has dimension {len(embeddings[0])}.")

    out_path = f"{rag.DATA_DIR}/04_embeddings.json"
    # Save chunks + embeddings together so stage 5 has everything it needs
    payload = [
        {**chunk, "embedding": emb}
        for chunk, emb in zip(chunks, embeddings)
    ]
    rag.save_json(payload, out_path)
    print(f"Saved chunk embeddings to {out_path}")


if __name__ == "__main__":
    main()
