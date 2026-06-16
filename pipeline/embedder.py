"""
embedder.py
-----------
Converts text chunks into dense vector embeddings using SentenceTransformers.

Model: all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Fast, lightweight, excellent for semantic similarity
  - Downloaded automatically on first use (~90 MB)
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Singleton model loader ────────────────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the model once and reuse it for the process lifetime."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model '{_MODEL_NAME}' (first-time download may take ~30s) …")
        _model = SentenceTransformer(_MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dim: {_model.get_sentence_embedding_dimension()}")
    return _model


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings and return a float32 numpy array of shape (N, 384).

    Args:
        texts:         List of strings to embed.
        batch_size:    Number of texts encoded per forward pass.
        show_progress: Show a tqdm progress bar for large batches.

    Returns:
        numpy.ndarray of shape (len(texts), embedding_dim), dtype float32.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress and len(texts) > 10,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim == dot product
    )
    return embeddings.astype(np.float32)


def embed_chunks(chunks: list[dict], **kwargs) -> tuple[np.ndarray, list[dict]]:
    """
    Embed a list of chunk dicts (produced by chunker.py).

    Args:
        chunks: List of {"chunk_id", "text", "source"} dicts.
        **kwargs: Forwarded to embed_texts (batch_size, show_progress).

    Returns:
        (embeddings, chunks) — the numpy array and the original chunk list
        (same order, so index i in embeddings matches chunks[i]).
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, **kwargs)
    print(f"[embedder] Embedded {len(chunks)} chunks → shape {embeddings.shape}")
    return embeddings, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        1-D numpy array of shape (384,), dtype float32.
    """
    return embed_texts([query], show_progress=False)[0]


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Paris is the capital of France.",
    ]
    vecs = embed_texts(sample_texts)
    print(f"Shape: {vecs.shape}")      # (3, 384)
    print(f"First vector norm: {np.linalg.norm(vecs[0]):.4f}")  # ~1.0 (L2-normalised)

    q = embed_query("What animal jumped?")
    sims = vecs @ q                    # cosine similarities (dot product on L2-normed vecs)
    for text, sim in zip(sample_texts, sims):
        print(f"  sim={sim:.3f}  {text}")