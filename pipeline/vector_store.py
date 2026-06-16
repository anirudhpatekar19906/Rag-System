"""
vector_store.py
---------------
Wraps a FAISS flat index for storing chunk embeddings and doing
fast nearest-neighbour similarity search.

Features:
  - Add embeddings + metadata in one call
  - Search by query vector → returns top-k chunks with scores
  - Save / load index + metadata to disk (no external DB required)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    In-memory FAISS index paired with a metadata list.

    Attributes:
        dim:       Embedding dimension (set on first add or from saved index).
        index:     faiss.IndexFlatIP — inner-product search on L2-normed vectors
                   is equivalent to cosine similarity.
        metadata:  List of chunk dicts aligned 1-to-1 with the FAISS vectors.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    # ── Insertion ─────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add embeddings and their corresponding chunk dicts to the store.

        Args:
            embeddings: float32 array of shape (N, dim).
            chunks:     List of N chunk dicts {"chunk_id", "text", "source"}.
        """
        assert embeddings.shape[0] == len(chunks), (
            f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"
        )
        assert embeddings.shape[1] == self.dim, (
            f"Embedding dim {embeddings.shape[1]} != store dim {self.dim}"
        )
        self.index.add(embeddings.astype(np.float32))
        self.metadata.extend(chunks)
        print(f"[vector_store] Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Find the top-k most similar chunks to a query vector.

        Args:
            query_vector: 1-D float32 array of shape (dim,).
            top_k:        Number of results to return.

        Returns:
            List of chunk dicts augmented with a "score" key (cosine similarity,
            higher = more relevant), sorted descending.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Vector store is empty — add documents first.")

        top_k = min(top_k, self.index.ntotal)
        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS returns -1 for missing results
                continue
            chunk = dict(self.metadata[idx])  # copy so we don't mutate stored metadata
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Persist the FAISS index and metadata to *directory*.

        Creates two files:
            <directory>/index.faiss
            <directory>/metadata.pkl
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, str(Path(directory) / "index.faiss"))
        with open(Path(directory) / "metadata.pkl", "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)
        print(f"[vector_store] Saved {self.index.ntotal} vectors to '{directory}'")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        """
        Load a previously saved VectorStore from *directory*.

        Returns:
            A fully initialised VectorStore instance.
        """
        index_path = Path(directory) / "index.faiss"
        meta_path = Path(directory) / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No saved index found in '{directory}'. "
                "Run ingestion first to build and save the store."
            )

        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)

        store = cls(dim=payload["dim"])
        store.index = index
        store.metadata = payload["metadata"]
        print(f"[vector_store] Loaded {store.index.ntotal} vectors from '{directory}'")
        return store

    # ── Helpers ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.dim}, vectors={self.index.ntotal})"


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    rng = np.random.default_rng(42)

    dim = 384
    store = VectorStore(dim=dim)

    # Fake embeddings + chunks
    fake_vecs = rng.standard_normal((5, dim)).astype(np.float32)
    # L2-normalise so inner product == cosine similarity
    fake_vecs /= np.linalg.norm(fake_vecs, axis=1, keepdims=True)

    fake_chunks = [{"chunk_id": i, "text": f"chunk {i}", "source": "test"} for i in range(5)]
    store.add(fake_vecs, fake_chunks)

    # Search
    q = fake_vecs[2]               # query == chunk 2, so it should be #1 result
    hits = store.search(q, top_k=3)
    print("Top results:")
    for h in hits:
        print(f"  score={h['score']:.4f}  text='{h['text']}'")

    # Save & reload
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        reloaded = VectorStore.load(tmpdir)
        hits2 = reloaded.search(q, top_k=3)
        assert hits2[0]["text"] == "chunk 2", "Reload test failed"
        print("Save / load round-trip: OK")