"""
03_chunking.py
---------------
Stage 3 of the RAG pipeline: split preprocessed documents into
overlapping text chunks suitable for embedding.

Run after 02_preprocessing.py:
    python 03_chunking.py
"""

import rag

def main():
    in_path = f"{rag.DATA_DIR}/02_preprocessed.json"
    documents = rag.load_json(in_path)

    chunks = rag.chunk_documents(documents)

    print(f"Created {len(chunks)} chunk(s) from {len(documents)} document(s).")
    print(f"  chunk_size={rag.CHUNK_SIZE}, overlap={rag.CHUNK_OVERLAP}")
    for c in chunks[:2]:
        preview = c["text"][:120].replace("\n", " ")
        print(f"  - {c['chunk_id']} ({c['source']}): {preview}...")

    out_path = f"{rag.DATA_DIR}/03_chunks.json"
    rag.save_json(chunks, out_path)
    print(f"\nSaved chunks to {out_path}")


if __name__ == "__main__":
    main()
