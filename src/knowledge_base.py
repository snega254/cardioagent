"""
Builds a retrieval index over the hand-authored knowledge base.

Default: TF-IDF + numpy cosine similarity — fully offline, zero downloads,
guaranteed to work regardless of internet access on the day of the deadline.

Optional upgrade: if you have time and internet, set USE_FAISS=True and it
will build a FAISS index over the same TF-IDF vectors (satisfies the
"vector database" requirement explicitly; TF-IDF vectors are valid vectors
for FAISS to index — FAISS does not require neural embeddings specifically).

Run standalone to sanity-check retrieval:
    python src/knowledge_base.py
"""
import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

USE_FAISS = True  # set False to skip FAISS entirely and use pure numpy


def load_kb(path="knowledge_base/kb_texts.json"):
    with open(path) as f:
        raw = json.load(f)
    entries = []  # list of (category, text)
    for category, texts in raw.items():
        for t in texts:
            entries.append((category, t))
    return entries


def build_index(kb_path="knowledge_base/kb_texts.json", out_path="kb_index.pkl"):
    entries = load_kb(kb_path)
    texts = [t for _, t in entries]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray().astype("float32")

    faiss_index = None
    if USE_FAISS:
        try:
            import faiss
            dim = vectors.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)  # inner product on L2-normalized vecs = cosine sim
            norm_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            faiss_index.add(norm_vectors)
        except ImportError:
            print("faiss not installed - falling back to numpy cosine similarity.")
            faiss_index = None

    with open(out_path, "wb") as f:
        pickle.dump({
            "entries": entries,
            "vectorizer": vectorizer,
            "vectors": vectors,
        }, f)

    if faiss_index is not None:
        import faiss
        faiss.write_index(faiss_index, "kb_index.faiss")

    print(f"Built index over {len(entries)} knowledge base entries.")
    print(f"FAISS index: {'built' if faiss_index is not None else 'not used (numpy fallback)'}")
    return out_path


if __name__ == "__main__":
    build_index()