# 🌐 DocRAG WebApp

**DocRAG WebApp** is a lightweight web interface built with **FastAPI** and a static frontend.  
It wraps around the [docrag-llm](https://pypi.org/project/docrag-llm) package, providing:  
- 📂 File and URL ingestion  
- 🧠 Storage in **ChromaDB** with Ollama embeddings  
- 🤖 Question-answering via local **Ollama LLMs**  
- ⚡ Streaming responses with status updates in the UI  

---

## ✨ Features
- Upload or link documents (`PDF`, `DOCX`, `PPTX`, `HTML`, etc.).
- Parse and chunk documents with [Docling](https://github.com/docling-project/docling).
- Store embeddings in [ChromaDB](https://www.trychroma.com/).
- Answer questions with [Ollama](https://ollama.ai/) (default model: `llama3.2:1b`).
- Interactive web UI with progress/status bar.

---

## 📦 Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.ai/)** running locally
  - Pull required models:  
    ```bash
    ollama pull llama3.2:1b --or any Ollam model
    ollama pull nomic-embed-text
    ```
- **Node/JS** only if you plan to rebuild the frontend (not required for static HTML/JS).

---

## 🚀 Run Locally

1. Clone or move into your `webapp` folder:
   ```bash
   cd webapp
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate   # macOS/Linux

   pip install -r requirements.txt
   ```

3. Start the server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open the web UI in your browser:
   ```
   http://localhost:8000
   ```

---

## 📂 Project Structure

```
webapp/
│── app.py              # FastAPI backend
│── static/             # Frontend (index.html, JS, CSS)
│── requirements.txt    # Python dependencies
│── README.md           # This file
```

---

## 🔧 API Endpoints

- `POST /api/ingest/url` → Ingest document from URL
- `POST /api/ingest/file` → Ingest uploaded file
- `POST /api/ask` → Ask a question (returns full answer)
- `POST /api/ask_stream` → Ask a question (streaming response)
- `GET /api/health` → Check service status

---

## 🖥️ Frontend

- Drag-and-drop file upload
- Input URL ingestion
- Question input box
- Real-time answer streaming
- Status bar for ingestion/query progress

---

## 🤝 Contributing

This project builds on the amazing work from:
- [**docrag-llm**](https://pypi.org/project/docrag-llm) (RAG pipeline)
- [**Docling**](https://github.com/docling-project/docling) (document parsing)
- [**ChromaDB**](https://www.trychroma.com/) (vector store)
- [**Ollama**](https://ollama.ai/) (local LLMs)

Contributions welcome!

---

## 📜 License

MIT © 2025  
