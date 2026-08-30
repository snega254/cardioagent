"""
Persistent vector store for CardioAgent's RAG knowledge base.

Uses real sentence embeddings (sentence-transformers, all-MiniLM-L6-v2 —
~80MB, runs fine on CPU with 8GB RAM) so that new documents can be added
incrementally without retraining anything (unlike TF-IDF, which needs the
vectorizer refit on the whole corpus every time you add a document).

This replaces the old hardcoded-15-entry TF-IDF index with a real,
growable, persistent store, per the project requirement. It stores which
source document each chunk came from, so retrieved evidence can always be
traced back to its source.
"""
import pickle

import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.chunks = []       # list of {"text": str, "source": str, "chunk_id": int}
        self.embeddings = None  # np.ndarray [N, dim], float32
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model '{EMBED_MODEL_NAME}' "
                  f"(first run downloads ~80MB, needs internet once)...")
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
        return self._model

    def _embed(self, texts):
        model = self._get_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def add_texts(self, texts, source):
        """Add a list of text chunks from one source document."""
        if not texts:
            return
        new_chunks = [{"text": t, "source": source, "chunk_id": i}
                      for i, t in enumerate(texts)]
        new_embeds = self._embed(texts)
        if self.embeddings is None:
            self.embeddings = new_embeds
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeds])
        self.chunks.extend(new_chunks)

    def search(self, query, top_k=3):
        """Returns list of (source, text, similarity_score), highest similarity first."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self._embed([query])[0]
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        q_norm = np.linalg.norm(query_vec) + 1e-8
        sims = (self.embeddings @ query_vec) / (norms * q_norm)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i]["source"], self.chunks[i]["text"], float(sims[i]))
                for i in top_idx]

    def list_documents(self):
        """Returns {source_name: chunk_count} — for the 'show indexed documents' UI requirement."""
        counts = {}
        for c in self.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return counts

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]