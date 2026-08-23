"""
RAG retrieval layer, backed by the persistent VectorStore (real document
embeddings) — not a hardcoded list.

IMPORTANT (enforces the required architectural separation): this module
ONLY retrieves text. It never receives ECG signal data — its only inputs
are text (a predicted class name and optional keywords). ECG analysis
happens entirely in model.py / gradcam.py.
"""
from vector_store import VectorStore


def retrieve(query_text, store_path="vector_store.pkl", top_k=3):
    store = VectorStore()
    store.load(store_path)
    return store.search(query_text, top_k=top_k)


def build_query(predicted_class, class_full_name, extra_keywords=None):
    query = f"{predicted_class} {class_full_name} ECG changes"
    if extra_keywords:
        query += " " + " ".join(extra_keywords)
    return query
