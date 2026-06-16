"""
chunker.py
----------
Splits raw text (or a list of page/segment dicts) into overlapping chunks
suitable for embedding.

Strategy: character-level sliding window with configurable size & overlap.
Each chunk carries its source metadata so we can cite it later.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Split *text* into overlapping windows and return a list of chunk dicts.

    Args:
        text:       The raw string to split.
        source:     Metadata label (e.g. "doc.pdf:page3") attached to every chunk.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of {"chunk_id": int, "text": str, "source": str} dicts.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap  # how far to advance each iteration

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_str,
                "source": source,
            })
            chunk_id += 1

        if end == len(text):
            break

        start += step

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Chunk a list of page/segment dicts produced by the loaders.

    Each dict is expected to have at least "text" and "source" keys.

    Returns:
        Flat list of chunk dicts with source inherited from the page/segment.
    """
    all_chunks: list[dict] = []

    # Special handling for YouTube: transcripts are often very short segments.
    # If the average segment length is small, we group multiple segments into
    # one "super-segment" before chunking to preserve conversational flow.
    is_youtube = any("yt:" in p.get("source", "") for p in pages)

    if is_youtube:
        # Group segments into larger blocks of text
        grouped_text = ""
        current_source = "unknown"

        # We group segments together until they hit chunk_size, then we treat it as a page
        temp_pages = []
        for page in pages:
            txt = page["text"]
            src = page.get("source", "unknown")

            if not grouped_text:
                current_source = src

            if len(grouped_text) + len(txt) < chunk_size:
                grouped_text += " " + txt
            else:
                temp_pages.append({"text": grouped_text.strip(), "source": current_source})
                grouped_text = txt
                current_source = src

        if grouped_text:
            temp_pages.append({"text": grouped_text.strip(), "source": current_source})

        pages = temp_pages

    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            source=page.get("source", "unknown"),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(page_chunks)

    # Re-assign globally unique chunk IDs
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    print(f"[chunker] Created {len(all_chunks)} chunks from {len(pages)} page(s)/segment(s)")
    return all_chunks


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "The quick brown fox jumps over the lazy dog. " * 30
    )
    chunks = chunk_text(sample, source="test", chunk_size=100, overlap=20)
    print(f"Total chunks: {len(chunks)}")
    for c in chunks[:4]:
        print(f"  [{c['chunk_id']}] {c['text'][:80]!r}")