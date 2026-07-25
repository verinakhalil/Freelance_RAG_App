"""
06_retrieve_context.py
------------------------
Stage 6 of the RAG pipeline: given a query, retrieve the most relevant
chunks from the Chroma vector store.

Run after 05_create_chroma_store.py:
    python 06_retrieve_context.py "your question here"
"""

import sys
import rag

def main():
    query = " ".join(sys.argv[1:]) or "What is this project about?"

    collection = rag.get_chroma_collection()
    contexts = rag.retrieve_context(query, k=rag.TOP_K, collection=collection)

    print(f"Query: {query}\n")
    if not contexts:
        print("No results found. Did you run the earlier pipeline stages?")
        return

    for i, c in enumerate(contexts, start=1):
        preview = c["text"][:150].replace("\n", " ")
        print(f"[{i}] source={c['source']} distance={c['distance']:.4f}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
