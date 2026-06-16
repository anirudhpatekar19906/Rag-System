import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from ingestion.pdf_loader import load_pdf
from ingestion.youtube_loader import load_youtube
from ingestion.github_loader import load_github
from pipeline.chunker import chunk_pages
from pipeline.embedder import embed_chunks, embed_query
from pipeline.vector_store import VectorStore
from llm.gemini_client import ask

class RAGService:
    """
    Singleton service that manages the VectorStore and coordinates
    the ingestion and query pipelines.
    """
    def __init__(self, index_dir: str = "my_pdf_index"):
        self.index_dir = index_dir
        self.dim = 384  # all-MiniLM-L6-v2 dimension
        self.store = self._initialize_store()

    def _initialize_store(self) -> VectorStore:
        # Load from disk if exists, otherwise create a new empty store
        if os.path.exists(self.index_dir):
            print(f"[RAGService] Loading existing index from {self.index_dir}...")
            return VectorStore.load(self.index_dir)

        print("[RAGService] Creating new VectorStore...")
        return VectorStore(dim=self.dim)

    def ingest(self, pdf_paths: List[str] = None, youtube_urls: List[str] = None, github_urls: List[str] = None):
        """
        Full pipeline: Load -> Chunk -> Embed -> Store.
        Runs synchronously to avoid blocking the FastAPI event loop.
        """
        all_pages = []

        def clean_path(path: str) -> str:
            t = path.strip()
            if len(t) >= 2 and ((t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'"))):
                return t[1:-1]
            return t

        # 1. Load Sources
        if pdf_paths:
            for path in pdf_paths:
                cleaned_path = clean_path(path)
                pages = load_pdf(cleaned_path)
                all_pages.extend(pages)

        if youtube_urls:
            for url in youtube_urls:
                cleaned_url = clean_path(url)
                segments = load_youtube(cleaned_url)
                all_pages.extend(segments)

        if github_urls:
            for url in github_urls:
                cleaned_url = clean_path(url)
                files = load_github(cleaned_url)
                all_pages.extend(files)

        if not all_pages:
            raise ValueError("No source content loaded. Please provide a valid PDF, YouTube URL, or GitHub URL.")

        # 2. Chunk - Using updated constants for better quality
        chunks = chunk_pages(all_pages, chunk_size=800, overlap=100)

        # 3. Embed
        embeddings, processed_chunks = embed_chunks(chunks)

        # 4. Store & Save
        self.store.add(embeddings, processed_chunks)
        self.store.save(self.index_dir)

        return {"chunks_added": len(processed_chunks), "total_chunks": len(self.store)}

        # 2. Chunk - Using updated constants for better quality
        chunks = chunk_pages(all_pages, chunk_size=800, overlap=100)

        # 3. Embed
        embeddings, processed_chunks = embed_chunks(chunks)

        # 4. Store & Save
        self.store.add(embeddings, processed_chunks)
        self.store.save(self.index_dir)

        return {"chunks_added": len(processed_chunks), "total_chunks": len(self.store)}

    async def query(self, question: str, top_k: int = 15) -> Dict[str, Any]:
        """
        Retrieval and Generation pipeline.
        """
        if len(self.store) == 0:
            raise ValueError("VectorStore is empty. Please ingest some data first.")

        # 1. Embed Query
        q_vec = embed_query(question)

        # 2. Search - Increased top_k for better context
        hits = self.store.search(q_vec, top_k=top_k)

        # 3. Generate Answer via Gemini
        answer = ask(question, hits)

        return {
            "answer": answer,
            "sources": hits  # Return the chunks used for citations
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "index_size": len(self.store),
            "index_path": self.index_dir
        }
