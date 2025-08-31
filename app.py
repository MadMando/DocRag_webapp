# app.py
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---- Docling
from docling.document_converter import DocumentConverter

# ---- Chroma (new client API)
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

# ---- LLM/Embeddings
import ollama

# =========================
# Config
# =========================
DEFAULT_PERSIST = os.getenv("DOCRAG_PERSIST", "./.chroma")
DEFAULT_COLLECTION = os.getenv("DOCRAG_COLLECTION", "web")
DEFAULT_EMBED = os.getenv("DOCRAG_EMBED", "nomic-embed-text")
DEFAULT_LLM = os.getenv("DOCRAG_LLM", "llama3.2:1b")
DEFAULT_CHUNK_CHARS = int(os.getenv("DOCRAG_CHUNK_CHARS", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DOCRAG_CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("DOCRAG_TOP_K", "2"))

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "static"  # must contain index.html

app = FastAPI(title="DocRAG Web API", version="1.2")

# CORS for local dev (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Startup: ensure frontend exists
# =========================
@app.on_event("startup")
def _ensure_frontend():
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        index.write_text(
            """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DocRAG Web</title>
<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:2rem}
code{background:#f4f4f4;padding:.2rem .4rem;border-radius:.25rem}</style>
</head><body>
<h1>DocRAG Web</h1>
<p>Frontend placeholder. Put your <code>index.html</code> here: <code>static/index.html</code>.</p>
<p>Try the API: <code>GET /api/health</code></p>
</body></html>""",
            encoding="utf-8",
        )


# =========================
# Helpers
# =========================
class OllamaEmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function that calls a local Ollama embedding model."""
    def __init__(self, model: str = DEFAULT_EMBED):
        self.model = model

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in inputs:
            resp = ollama.embeddings(model=self.model, prompt=text)
            out.append(resp["embedding"])
        return out


def chunk_text(text: str, target_chars: int, overlap: int) -> List[str]:
    """Sentence-aware chunking with overlap."""
    sentences = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for s in sentences:
        s_len = len(s)
        if size + s_len + 1 > target_chars and buf:
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                buf = [tail, s]
                size = len(tail) + s_len
            else:
                buf = [s]
                size = s_len
        else:
            buf.append(s)
            size += s_len + 1
    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)
    if not chunks:
        # Fallback
        chunks = [text[i:i + target_chars] for i in range(0, len(text), target_chars)]
    return chunks


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()  # deterministic IDs


def _chroma_collection(persist: str, collection: str, embed_model: str):
    os.makedirs(persist, exist_ok=True)
    client = PersistentClient(path=persist)
    ef = OllamaEmbeddingFunction(model=embed_model)
    col = client.get_or_create_collection(name=collection, embedding_function=ef)
    return col


def _docling_convert(source: str) -> str:
    conv = DocumentConverter()
    res = conv.convert(source)
    md = res.document.export_to_markdown()
    if not md or not md.strip():
        raise RuntimeError("Docling returned empty text.")
    return md


def _index_chunks(col, chunks: List[str], source_tag: str) -> int:
    # Deterministic IDs so re-ingest won’t duplicate
    ids = [f"{source_tag}_{i:06d}" for i in range(len(chunks))]
    # Best-effort delete first (ignore if not present)
    try:
        col.delete(ids=ids)
    except Exception:
        pass
    col.add(documents=chunks, ids=ids, metadatas=[{"source": source_tag}] * len(chunks))
    return len(chunks)


def _sse(payload: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _retrieve(col, query: str, top_k: int) -> List[str]:
    res = col.query(query_texts=[query], n_results=max(1, top_k))
    docs_nested = (res.get("documents") or [[]])[0] if isinstance(res, dict) else []
    return list(docs_nested)


# =========================
# API
# =========================

import os
import logging
import ollama
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# --- Configuration ---
# Use an environment variable for the host, with a sensible default.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Configure logging to see output in your terminal.
logging.basicConfig(level=logging.INFO)

# Create a single, explicit client for the app to use.
app = FastAPI()
client = ollama.Client(host=OLLAMA_HOST)


@app.get("/api/tags")  # Renamed endpoint
def list_ollama_tags() -> JSONResponse:
    """Return available Ollama models (tags)."""
    try:
        logging.info(f"Attempting to list models from Ollama at {OLLAMA_HOST}...")
        
        # Use the configured client to make the call.
        data = client.list()
        
        # This logic correctly parses the response from client.list().
        names = [m.get("name") for m in data.get("models", []) if m.get("name")]
        logging.info(f"Successfully retrieved models: {names}")
        
        return JSONResponse({"ok": True, "models": names})

    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": "Could not connect to Ollama service."}, status_code=500)

# @app.get("/api/models")
# def list_models() -> JSONResponse:
#     """Return available Ollama models (for a UI dropdown)."""
#     try:
#         data = ollama.list()  # {'models': [{'name': 'llama3.2:1b', ...}, ...]}
#         names = [m.get("name") for m in data.get("models", []) if m.get("name")]
#         return JSONResponse({"ok": True, "models": names})
#     except Exception as e:
#         return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ---------- Ingest (non-streaming JSON) ----------
@app.post("/api/ingest/url")
def ingest_url(
    url: str = Form(...),
    persist: str = Form(DEFAULT_PERSIST),
    collection: str = Form(DEFAULT_COLLECTION),
    embed_model: str = Form(DEFAULT_EMBED),
    chunk_chars: int = Form(DEFAULT_CHUNK_CHARS),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> Dict[str, Any]:
    try:
        text = _docling_convert(url)
        chunks = chunk_text(text, chunk_chars, chunk_overlap)
        col = _chroma_collection(persist, collection, embed_model)
        added = _index_chunks(col, chunks, source_tag=_hash_id(url))
        return {"ok": True, "chunks_added": added, "source": url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    persist: str = Form(DEFAULT_PERSIST),
    collection: str = Form(DEFAULT_COLLECTION),
    embed_model: str = Form(DEFAULT_EMBED),
    chunk_chars: int = Form(DEFAULT_CHUNK_CHARS),
    chunk_overlap: int = Form(DEFAULT_CHUNK_OVERLAP),
) -> Dict[str, Any]:
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        text = _docling_convert(tmp_path)
        chunks = chunk_text(text, chunk_chars, chunk_overlap)
        col = _chroma_collection(persist, collection, embed_model)
        added = _index_chunks(col, chunks, source_tag=_hash_id(file.filename or tmp_path))
        return {"ok": True, "chunks_added": added, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ---------- Ingest (SSE progress for status bar) ----------
@app.get("/api/ingest/url_stream")
def ingest_url_stream(
    url: str,
    persist: str = DEFAULT_PERSIST,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    def gen() -> Generator[bytes, None, None]:
        try:
            yield _sse({"phase": "start", "progress": 0, "msg": "Starting"})
            yield _sse({"phase": "convert", "progress": 20, "msg": "Converting with Docling"})
            text = _docling_convert(url)
            yield _sse({"phase": "chunk", "progress": 40, "msg": "Chunking text"})
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            yield _sse({"phase": "index", "progress": 70, "msg": "Indexing into Chroma"})
            col = _chroma_collection(persist, collection, embed_model)
            added = _index_chunks(col, chunks, source_tag=_hash_id(url))
            yield _sse({"phase": "done", "progress": 100, "ok": True, "chunks_added": added})
        except Exception as e:
            yield _sse({"phase": "error", "progress": 100, "ok": False, "error": str(e)})
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/ingest/file_stream")
async def ingest_file_stream(
    file: UploadFile = File(...),
    persist: str = DEFAULT_PERSIST,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    def gen() -> Generator[bytes, None, None]:
        try:
            yield _sse({"phase": "start", "progress": 0, "msg": "Starting"})
            yield _sse({"phase": "convert", "progress": 20, "msg": f"Converting {file.filename}"})
            text = _docling_convert(tmp_path)
            yield _sse({"phase": "chunk", "progress": 40, "msg": "Chunking text"})
            chunks = chunk_text(text, chunk_chars, chunk_overlap)
            yield _sse({"phase": "index", "progress": 70, "msg": "Indexing into Chroma"})
            col = _chroma_collection(persist, collection, embed_model)
            added = _index_chunks(col, chunks, source_tag=_hash_id(file.filename or tmp_path))
            yield _sse({"phase": "done", "progress": 100, "ok": True, "chunks_added": added})
        except Exception as e:
            yield _sse({"phase": "error", "progress": 100, "ok": False, "error": str(e)})
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- Ask (JSON) ----------
@app.post("/api/ask")
def ask(
    question: str = Form(...),
    persist: str = Form(DEFAULT_PERSIST),
    collection: str = Form(DEFAULT_COLLECTION),
    llm_model: str = Form(DEFAULT_LLM),
    embed_model: str = Form(DEFAULT_EMBED),
    top_k: int = Form(DEFAULT_TOP_K),
) -> Dict[str, Any]:
    try:
        col = _chroma_collection(persist, collection, embed_model)
        docs = _retrieve(col, question, top_k=top_k)
        context = "\n\n".join(docs) if docs else ""
        prompt = (
            "Use ONLY the provided context to answer. If insufficient, say so briefly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        )
        reply = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
        return {"ok": True, "answer": reply["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- Ask (streaming tokens; text/plain) ----------
@app.post("/api/ask_stream")
def ask_stream(
    question: str = Form(...),
    persist: str = Form(DEFAULT_PERSIST),
    collection: str = Form(DEFAULT_COLLECTION),
    llm_model: str = Form(DEFAULT_LLM),
    embed_model: str = Form(DEFAULT_EMBED),
    top_k: int = Form(DEFAULT_TOP_K),
):
    def gen() -> Generator[bytes, None, None]:
        try:
            col = _chroma_collection(persist, collection, embed_model)
            docs = _retrieve(col, question, top_k=top_k)
            context = "\n\n".join(docs) if docs else ""
            prompt = (
                "Use ONLY the provided context to answer. If insufficient, say so briefly.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
            )
            for chunk in ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                part = chunk.get("message", {}).get("content", "")
                if part:
                    yield part.encode("utf-8")
        except Exception as e:
            yield f"\n[stream-error] {e}".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/plain")


# ---------- Static frontend at root ----------
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------- Favicon (silence 404s) ----------
@app.get("/favicon.ico")
def favicon():
    fav = FRONTEND_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(fav)
    return HTMLResponse(status_code=204, content="")
