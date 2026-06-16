"""
main.py
-------
Universal RAG pipeline — terminal Q&A loop.

Usage examples:
    python main.py --pdf path/to/doc.pdf
    python main.py --youtube "https://youtu.be/dQw4w9WgXcQ"
    python main.py --github "https://github.com/user/repo"
    python main.py --pdf doc.pdf --youtube "https://youtu.be/..." --github "https://github.com/user/repo"

    # Re-use a previously built index (skip ingestion):
    python main.py --load-index ./saved_index

    # Ingest and save index for later:
    python main.py --pdf doc.pdf --save-index ./saved_index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env first so GEMINI_API_KEY is available ────────────────────────────
load_dotenv()

# ── Internal modules ──────────────────────────────────────────────────────────
from ingestion.pdf_loader import load_pdf
from ingestion.youtube_loader import load_youtube
from ingestion.github_loader import load_github
from pipeline.chunker import chunk_pages
from pipeline.embedder import embed_chunks, embed_query
from pipeline.vector_store import VectorStore
from llm.gemini_client import ask

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800          # Increased from 500 for better context
CHUNK_OVERLAP = 100       # Increased from 50 to maintain flow between chunks
TOP_K = 15                # Increased from 5 to retrieve more context for LLM
EMBEDDING_DIM = 384       # all-MiniLM-L6-v2 output dimension


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def collect_and_process_sources(store: VectorStore | None = None) -> VectorStore:
    """
    A smart loop that detects the source type (PDF, YouTube, GitHub)
    and processes it immediately.
    """
    print("\n--- 📥 Smart Source Ingestion ---")
    print("Enter a PDF path or a Link (YouTube/GitHub). Type 'done' to finish.")
    print("Tip: You can paste paths with quotes; they will be automatically removed.")

    while True:
        raw_input = input("👉 Source: ").strip()
        if not raw_input or raw_input.lower() == 'done':
            break

        # Clean surrounding quotes
        text = raw_input
        if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
            text = text[1:-1]

        # Simple type detection
        pdf_paths, yt_urls, gh_urls = [], [], []
        if text.lower().endswith('.pdf'):
            pdf_paths.append(text)
        elif 'youtube.com' in text or 'youtu.be' in text:
            yt_urls.append(text)
        elif 'github.com' in text:
            gh_urls.append(text)
        else:
            print("⚠️  Unknown source type. Please provide a .pdf path or a valid URL.")
            continue

        # Process immediately for immediate feedback
        store = ingest_sources(pdf_paths, yt_urls, gh_urls, store=store)
        print(f"✅ Processed successfully. Current index size: {len(store)} chunks")

    return store


def ingest_sources(

    pdf_paths: list[str],
    youtube_urls: list[str],
    github_urls: list[str],
    store: VectorStore | None = None,
) -> VectorStore:
    """
    Load all sources, chunk them, embed, and store in a VectorStore.

    Returns a populated VectorStore ready for querying.
    """
    all_pages: list[dict] = []

    # PDFs
    for pdf_path in pdf_paths:
        print(f"\n📄  Ingesting PDF: {pdf_path}")
        pages = load_pdf(pdf_path)
        all_pages.extend(pages)

    # YouTube videos
    for url in youtube_urls:
        print(f"\n🎬  Ingesting YouTube: {url}")
        segments = load_youtube(url)
        all_pages.extend(segments)

    # GitHub repos
    for url in github_urls:
        print(f"\n🐙  Ingesting GitHub: {url}")
        files = load_github(url)
        all_pages.extend(files)

    if not all_pages:
        print("[main] No source content loaded — nothing to ingest.")
        # If we are updating an existing store, we shouldn't exit
        if store is None:
            sys.exit(1)
        return store

    # Chunk
    print("\n✂️   Chunking text …")
    chunks = chunk_pages(all_pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Embed
    print("\n🔢  Generating embeddings …")
    embeddings, chunks = embed_chunks(chunks)

    # Store
    if store is None:
        store = VectorStore(dim=EMBEDDING_DIM)

    store.add(embeddings, chunks)

    return store


# ── Q&A loop ──────────────────────────────────────────────────────────────────

def qa_loop(store: VectorStore) -> None:
    """
    Interactive terminal loop: user types a question, gets an answer.
    Type 'quit' or 'exit' (or Ctrl+C) to stop.
    """
    print("\n" + "=" * 60)
    print("  RAG Q&A ready!  (type 'quit' to exit)")
    print(f"  Index size: {len(store)} chunks")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("❓  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        # Interactive Ingestion Command
        if query.startswith("/ingest"):
            print("\n🚀 Starting incremental ingestion...")
            store = collect_and_process_sources(store=store)
            print(f"\n✅ Ingestion sequence complete. Total index size: {len(store)} chunks")
            continue

        # Retrieve
        print("\n🔍  Retrieving relevant chunks …")
        q_vec = embed_query(query)
        hits = store.search(q_vec, top_k=TOP_K)

        # Show sources
        print(f"   Found {len(hits)} relevant chunk(s):")
        for h in hits:
            print(f"     score={h['score']:.3f}  source={h['source']}")

        # Generate
        print("\n🤖  Asking Gemini …\n")
        answer = ask(query, hits)

        print("─" * 60)
        print(answer)
        print("─" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Universal RAG — ingest PDFs, YouTube videos, or GitHub repos and ask questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--pdf", metavar="PATH", action="append", default=[],
                   help="Path to a PDF file (repeatable).")
    p.add_argument("--youtube", metavar="URL", action="append", default=[],
                   help="YouTube URL or video ID (repeatable).")
    p.add_argument("--github", metavar="URL", action="append", default=[],
                   help="GitHub repo URL or local path (repeatable).")
    p.add_argument("--save-index", metavar="DIR",
                   help="After ingestion, save the FAISS index to this directory.")
    p.add_argument("--load-index", metavar="DIR",
                   help="Skip ingestion and load a previously saved index.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.load_index:
        # ── Load existing index ────────────────────────────────────────────
        print(f"\n📂  Loading saved index from '{args.load_index}' …")
        store = VectorStore.load(args.load_index)
    else:
        # ── Ingest from sources (Automatic or Interactive) ──────────────────
        if any([args.pdf, args.youtube, args.github]):
            # Automatic mode: arguments provided
            store = ingest_sources(
                pdf_paths=args.pdf,
                youtube_urls=args.youtube,
                github_urls=args.github,
            )
            if args.save_index:
                store.save(args.save_index)
        else:
            # Interactive mode: no arguments provided
            print("\n" + "═" * 60)
            print("  Welcome to Multi-RAG AI!  ")
            print("═" * 60)
            print("1. 📥 Ingest new sources (PDFs, YouTube, GitHub)")
            print("2. 📂 Load an existing index")
            print("3. 🚪 Exit")

            choice = input("\nSelect an option (1-3): ").strip()

            if choice == "1":
                store = collect_and_process_sources()
                if store is None:
                    print("\n⚠️  No sources were processed. Exiting.")
                    sys.exit(0)

                save_choice = input("\n💾 Would you like to save this index to disk? (y/n): ").lower().strip()
                if save_choice == 'y':
                    save_dir = input("Enter directory path to save index: ").strip()
                    if save_dir:
                        store.save(save_dir)

            elif choice == "2":
                save_dir = input("\n📂 Enter the directory path of the saved index: ").strip()
                if not save_dir:
                    print("\n⚠️  No directory provided. Exiting.")
                    sys.exit(0)
                try:
                    store = VectorStore.load(save_dir)
                except Exception as e:
                    print(f"\n❌ Error loading index: {e}")
                    sys.exit(1)

            elif choice == "3":
                print("Goodbye!")
                sys.exit(0)
            else:
                print("\n⚠️  Invalid choice. Exiting.")
                sys.exit(0)

    qa_loop(store)


if __name__ == "__main__":
    main()