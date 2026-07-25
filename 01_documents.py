"""
01_documents.py
----------------
Stage 1 of the RAG pipeline: load raw documents.

Put your .txt source documents in data/raw/ before running this script.

Run:
    python 01_documents.py
"""

import rag

def main():
    documents = rag.load_documents()

    if not documents:
        print(f"No .txt files found in {rag.RAW_DOCS_DIR}")
        print("Add some .txt documents there and re-run this script.")
        return

    print(f"Loaded {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} characters)")

    out_path = f"{rag.DATA_DIR}/01_documents.json"
    rag.save_json(documents, out_path)
    print(f"\nSaved raw documents to {out_path}")


if __name__ == "__main__":
    main()
