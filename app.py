# app.py
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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
DEFAULT_PERSIST = "./.chroma"
DEFAULT_COLLECTION = "web"
DEFAULT_EMBED = "nomic-embed-text"
DEFAULT_LLM = "llama3.2:1b"
DEFAULT_CHUNK_CHARS = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 2

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "static"  # must contain index.html

app = FastAPI(title="DocRAG Web API", version="1.1")

# CORS for local dev (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _index_chunks(
    col,
    chunks: List[str],
    source_tag: str,
) -> int:
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
@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "frontend_dir": str(FRONTEND_DIR)})


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
