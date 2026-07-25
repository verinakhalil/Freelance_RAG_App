"""
02_preprocessing.py
--------------------
Stage 2 of the RAG pipeline: clean the raw document text
(normalize whitespace, strip control characters, etc.).

Run after 01_documents.py:
    python 02_preprocessing.py
"""

import rag

def main():
    in_path = f"{rag.DATA_DIR}/01_documents.json"
    documents = rag.load_json(in_path)

    documents = rag.preprocess_documents(documents)

    print(f"Preprocessed {len(documents)} document(s).")
    for doc in documents[:1]:
        preview = doc["text"][:200].replace("\n", " ")
        print(f"  Example preview ({doc['source']}): {preview}...")

    out_path = f"{rag.DATA_DIR}/02_preprocessed.json"
    rag.save_json(documents, out_path)
    print(f"\nSaved preprocessed documents to {out_path}")


if __name__ == "__main__":
    main()
