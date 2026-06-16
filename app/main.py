import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from app.services.rag_service import RAGService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-rag-api")

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Multi-RAG API", description="AI-powered document Q&A")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the RAG Service as a singleton
# Using the same index folder as the CLI tool for consistency
rag_service = RAGService(index_dir="my_pdf_index")

# ── Request/Response Models ────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 15

class IngestRequest(BaseModel):
    youtube_url: Optional[str] = None
    github_url: Optional[str] = None

# ── Endpoints ──────────────────────────────────────────────────────────────────

def run_ingestion_task(pdf_paths=None, youtube_urls=None, github_urls=None):
    """
    Background task to handle the heavy lifting of ingestion.
    """
    try:
        logger.info(f"Background ingestion started: PDFs={pdf_paths}, YT={youtube_urls}, GH={github_urls}")
        rag_service.ingest(pdf_paths=pdf_paths, youtube_urls=youtube_urls, github_urls=github_urls)
        logger.info("Background ingestion completed successfully.")
    except Exception as e:
        logger.exception(f"Background ingestion failed: {e}")
    finally:
        # Clean up any temporary PDF files created during the request
        if pdf_paths:
            for path in pdf_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        logger.info(f"Cleaned up temp file: {path}")
                except Exception as cleanup_err:
                    logger.error(f"Could not remove temporary file {path}: {cleanup_err}")

@app.get("/status")
async def get_status():
    """Returns current index status."""
    return rag_service.get_status()

@app.post("/ingest-link")
async def ingest_link(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest a YouTube or GitHub link in the background."""
    try:
        youtube_urls = [request.youtube_url] if request.youtube_url else []
        github_urls = [request.github_url] if request.github_url else []

        background_tasks.add_task(
            run_ingestion_task,
            youtube_urls=youtube_urls,
            github_urls=github_urls
        )
        return {"message": "Ingestion started in background. Please check status in a few moments."}
    except Exception as e:
        logger.exception("Failed to queue link ingestion")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ingest-pdf")
async def ingest_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF and ingest it in the background."""
    temp_path = None
    try:
        unique_filename = f"temp_{uuid.uuid4().hex}.pdf"
        temp_path = Path(unique_filename)

        logger.info(f"Uploading file: {file.filename} as {temp_path}")

        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        background_tasks.add_task(
            run_ingestion_task,
            pdf_paths=[str(temp_path)]
        )

        return {"message": "PDF uploaded and ingestion started in background."}
    except Exception as e:
        logger.exception(f"Failed to upload PDF {file.filename}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Ask a question based on the ingested knowledge."""
    try:
        result = await rag_service.query(request.question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
