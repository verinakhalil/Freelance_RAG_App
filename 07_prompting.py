"""
07_prompting.py
-----------------
Stage 7 of the RAG pipeline: build a grounded prompt from retrieved
context and generate an answer via the OpenRouter API.

Requires OPENROUTER_API_KEY to be set as an environment variable, e.g.:
    export OPENROUTER_API_KEY="your_key_here"     (Mac/Linux)
    setx OPENROUTER_API_KEY "your_key_here"        (Windows)

Never hard-code the real key in this file.

Run after 05_create_chroma_store.py:
    python 07_prompting.py "your question here"
"""

import sys
import rag

def main():
    query = " ".join(sys.argv[1:]) or "What is this project about?"

    collection = rag.get_chroma_collection()
    contexts = rag.retrieve_context(query, k=rag.TOP_K, collection=collection)

    if not contexts:
        print("No context retrieved. Did you run the earlier pipeline stages?")
        return

    answer = rag.generate_answer(query, contexts)

    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}\n")

    print("Sources used:")
    for c in contexts:
        print(f"  - {c['source']}")


if __name__ == "__main__":
    main()
