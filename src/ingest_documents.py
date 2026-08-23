"""
Ingests documents (PDF/TXT) and/or the small built-in knowledge base into
the persistent vector store.

Document -> text extraction -> cleaning -> chunking -> embeddings -> store

Run:
    python src/ingest_documents.py --seed_default_kb
    python src/ingest_documents.py --docs_dir documents --store_path vector_store.pkl
"""
import argparse
import json
import os

from document_processing import chunk_text, clean_text, extract_text
from vector_store import VectorStore


def seed_default_kb(store, kb_path="knowledge_base/kb_texts.json"):
    """Adds the small, hand-written built-in ECG knowledge base to `store`.
    Returns True if the file was found and added, False otherwise."""
    if not os.path.exists(kb_path):
        return False
    with open(kb_path) as f:
        raw = json.load(f)
    for category, texts in raw.items():
        store.add_texts(texts, source=f"builtin:{category}")
    return True


def ingest_folder(store, docs_dir):
    """Ingests all .pdf/.txt files in docs_dir into `store`. Returns a
    list of (filename, n_chunks) for files successfully added."""
    added = []
    if not os.path.isdir(docs_dir):
        return added
    files = [f for f in os.listdir(docs_dir) if f.lower().endswith((".pdf", ".txt"))]
    for fname in files:
        path = os.path.join(docs_dir, fname)
        try:
            raw_text = extract_text(path)
        except Exception as e:
            print(f"  ERROR extracting {fname}: {e} - skipping.")
            continue
        cleaned = clean_text(raw_text)
        if not cleaned:
            print(f"  WARNING: no extractable text in {fname} - skipping.")
            continue
        chunks = chunk_text(cleaned)
        store.add_texts(chunks, source=fname)
        added.append((fname, len(chunks)))
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="documents")
    parser.add_argument("--store_path", default="vector_store.pkl")
    parser.add_argument("--seed_default_kb", action="store_true")
    args = parser.parse_args()

    store = VectorStore()
    if os.path.exists(args.store_path):
        print(f"Found existing store at {args.store_path} - loading and adding to it.")
        store.load(args.store_path)
    else:
        print("No existing store found - creating a new one.")

    if args.seed_default_kb:
        ok = seed_default_kb(store)
        print("Seeded default hand-written knowledge base entries."
              if ok else "WARNING: default knowledge base file not found, skipped.")

    added = ingest_folder(store, args.docs_dir)
    for fname, n_chunks in added:
        print(f"Added {n_chunks} chunks from {fname}")
    if not added and not os.path.isdir(args.docs_dir):
        print(f"Documents directory '{args.docs_dir}' not found - "
              f"create it and add .pdf/.txt files, or pass --docs_dir.")

    store.save(args.store_path)
    print(f"\nSaved vector store to {args.store_path}")
    print("Documents currently indexed:")
    docs = store.list_documents()
    for src, count in (docs.items() if docs else []):
        print(f"  {src}: {count} chunks")
    print(f"\nTotal chunks in index: {len(store.chunks)}")


if __name__ == "__main__":
    main()
