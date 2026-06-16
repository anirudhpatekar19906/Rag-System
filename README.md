# Multi-RAG AI 🤖

Multi-RAG AI is a universal Retrieval-Augmented Generation (RAG) system that allows you to ingest knowledge from diverse sources—PDFs, YouTube videos, and GitHub repositories—and query that knowledge using a Large Language Model (LLM).

## ✨ Features

- **Multi-Source Ingestion**: 
  - 📄 **PDFs**: Extract text from local PDF documents.
  - 🎬 **YouTube**: Transcribe and ingest content from YouTube videos.
  - 🐙 **GitHub**: Load and index source code and documentation from GitHub repositories.
- **Advanced RAG Pipeline**:
  - **Smart Chunking**: Optimized text splitting to maintain context.
  - **Efficient Embedding**: Uses `all-MiniLM-L6-v2` for high-quality vector representations.
  - **Vector Storage**: Powered by **FAISS** for lightning-fast similarity search.
- **LLM Powered Answers**: Integration with **Google Gemini** to provide context-aware, accurate answers based on retrieved data.
- **Flexible Interfaces**:
  - **Interactive CLI**: A powerful terminal loop for quick ingestion and Q&A.
  - **Web Frontend**: A modern user interface for a seamless experience.
- **Index Persistence**: Save your processed knowledge bases to disk and reload them instantly.

## 🏗️ Architecture

The project is organized into modular components:

- `ingestion/`: Specialized loaders for different data formats.
- `pipeline/`: Core logic for chunking, embedding, and vector store management.
- `llm/`: Interface for communicating with the LLM (Gemini).
- `app/`: Backend services and API coordination.
- `frontend/`: React-based user interface.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Gemini API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Multi_Rag
   ```

2. **Set up environment variables**:
   Create a `.env` file in the root directory and add your API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠️ Usage

### Command Line Interface (CLI)

You can start the RAG system directly from the terminal.

**Automatic Ingestion:**
```bash
python main.py --pdf path/to/document.pdf --youtube "https://youtube.com/..." --github "https://github.com/user/repo"
```

**Load Existing Index:**
```bash
python main.py --load-index ./my_saved_index
```

**Save Index After Ingestion:**
```bash
python main.py --pdf doc.pdf --save-index ./my_saved_index
```

**Interactive Mode:**
Just run `python main.py` and follow the on-screen prompts to ingest data or load an index. Once in the Q&A loop, you can use `/ingest` to add more sources on the fly.

### Web Interface

The project includes a frontend. You can launch the backend and frontend separately as per the project's deployment instructions (typically via `start_rag.bat` or running the `app/main.py` and the Vite dev server).

## ⚙️ Configuration

You can adjust the RAG parameters in `main.py` or the service configuration:
- `CHUNK_SIZE`: Controls the size of text segments.
- `CHUNK_OVERLAP`: Maintains context between segments.
- `TOP_K`: Number of relevant chunks retrieved for the LLM.

## 🎯 Example Use-Cases

Multi-RAG AI is designed for cross-modal knowledge synthesis. For example, you can:

- **Research a new library**: Ingest the library's official **PDF documentation**, a **YouTube tutorial** on its implementation, and the **GitHub repository** containing the source code. You can then ask: *"Based on the tutorial and the source code, how do I implement a custom middleware in this library?"*
- **Study for Exams**: Ingest several **textbooks (PDF)** and **lecture videos (YouTube)**. Query: *"Compare the theory explained in the PDF with the practical example shown in the lecture video."*
- **Analyze Repos**: Ingest a complex **GitHub repo** to understand the architecture. Query: *"Which files handle the database connection and how is the connection pooling managed?"*

## ⚠️ Known Limitations

- **PDF Layouts**: Complex PDFs with multi-column layouts or heavy imagery may result in noisy text extraction.
- **Embedding Model**: `all-MiniLM-L6-v2` is efficient but may struggle with extremely niche technical jargon compared to larger models.
- **Context Window**: While Gemini has a large window, retrieving too many `TOP_K` chunks may occasionally lead to "lost in the middle" phenomena.

## 🗺️ Future Roadmap

- [ ] **Hybrid Search**: Combine vector search with keyword-based BM25 search for better accuracy.
- [ ] **Multi-LLM Support**: Allow users to switch between Gemini, OpenAI, and Claude.
- [ ] **Advanced PDF Processing**: Integration with OCR tools like Tesseract or LayoutLM for better structure extraction.
- [ ] **Citation Mapping**: Highlighting the exact page/timestamp/line in the UI for every answer.
- [ ] **Real-time Web Crawling**: Ingest dynamic websites and blogs on the fly.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the pipeline:
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✍️ About the Author

**Anirudh Patekar**

A passionate developer focused on building intelligent systems that bridge the gap between unstructured data and actionable knowledge. 

- GitHub: [@anirudhpatekar](https://github.com/anirudhpatekar)
- LinkedIn: [Anirudh Patekar](https://www.linkedin.com/in/anirudh-patekar-1586b5336)
