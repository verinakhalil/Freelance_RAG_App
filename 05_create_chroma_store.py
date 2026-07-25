"""
05_create_chroma_store.py
---------------------------
Stage 5 of the RAG pipeline: load chunk embeddings into a persistent
Chroma vector store on disk.

Run after 04_vector_representation.py:
    python 05_create_chroma_store.py
"""

import rag

def main():
    in_path = f"{rag.DATA_DIR}/04_embeddings.json"
    items = rag.load_json(in_path)

    chunks = [{"chunk_id": i["chunk_id"], "source": i["source"], "text": i["text"]} for i in items]
    embeddings = [i["embedding"] for i in items]

    collection = rag.get_chroma_collection()
    rag.add_chunks_to_store(chunks, embeddings, collection=collection)

    print(f"Stored {len(chunks)} chunk(s) in Chroma collection "
          f"'{rag.COLLECTION_NAME}' at {rag.CHROMA_DIR}")


if __name__ == "__main__":
    main()
