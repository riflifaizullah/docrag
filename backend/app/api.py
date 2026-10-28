"""FastAPI app: /ask (single- and multi-question), /sync, /health,
/documents (listing, preview, download), /monitor (status page)."""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Optional

import pdfplumber
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app import config, database, generation, lang, retrieval, sync_documents
from app.lang import ui

app = FastAPI(title="DocRAG API")

# Only requests addressed to this machine are served, which blocks DNS rebinding from a web page.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("DOCRAG_ALLOWED_HOSTS", "localhost,127.0.0.1").split(","),
)

PREVIEW_TEXT_CHARS = 800
THUMBNAIL_DPI = 100


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    answered: bool = False
    refused: bool = False
    needs_clarification: bool = False
    candidate_documents: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    log_id: Optional[int] = Field(
        default=None,
        description="answer_log row id for this sub-answer, or null when the "
        "turn has no single log entry to flag (e.g. a multi-document combined "
        "answer). Pass to POST /flag/{log_id} to record a human review flag.",
    )


class AskResponse(BaseModel):
    results: list[QuestionAnswer]
    answered: bool = False
    refused: bool = False
    needs_clarification: bool = False
    candidate_documents: list[str] = Field(default_factory=list)


@app.on_event("startup")
def on_startup():
    database.init_db()
    sync_documents.start_scheduler()


@app.on_event("shutdown")
def on_shutdown():
    sync_documents.stop_scheduler()


@app.get("/ui")
def ui_strings():
    return {"language": "en", "strings": ui.STRINGS}


@app.get("/health")
def health():
    return {"status": "ok", "model": config.OLLAMA_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    session_id = req.session_id or database.create_session()
    database.save_message(session_id, "user", req.question)

    sub_results = retrieval.run_on_chroma_thread(
        generation.generate_multi_answer, req.question, session_id=session_id
    )
    results = [QuestionAnswer(**r) for r in sub_results]

    combined_answer = "\n\n".join(r.answer for r in results)
    database.save_message(session_id, "assistant", combined_answer)

    return AskResponse(
        results=results,
        answered=all(r.answered for r in results) if results else False,
        refused=any(r.refused for r in results),
        needs_clarification=any(r.needs_clarification for r in results),
        candidate_documents=[c for r in results for c in r.candidate_documents],
    )


@app.post("/sync")
def sync():
    return retrieval.run_on_chroma_thread(sync_documents.sync_once)


@app.get("/sources")
def sources():
    return {"sources": retrieval.list_indexed_sources()}


def _validated_path(filename: str):
    """Filename must be one of the actually-indexed sources -- never a raw
    path lookup, to avoid path traversal onto arbitrary local files."""
    if filename not in retrieval.list_indexed_sources():
        raise HTTPException(status_code=404, detail="Document not found in index")
    path = config.CORPUS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Document file missing on disk")
    return path


@app.get("/documents")
def documents():
    sources = retrieval.list_indexed_sources()
    labels = retrieval.run_on_chroma_thread(retrieval.get_document_type_labels, sources)
    cached_meta = database.get_document_metadata(sources)
    results = []
    for source in sources:
        path = config.CORPUS_DIR / source
        if not path.is_file():
            continue
        if source in cached_meta:
            page_count, size_kb = cached_meta[source]
        else:
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
            size_kb = round(path.stat().st_size / 1024, 1)
            database.set_document_metadata(source, page_count, size_kb)
        results.append(
            {
                "filename": source,
                "type": labels.get(source, lang.get().UNKNOWN_TYPE),
                "page_count": page_count,
                "size_kb": size_kb,
            }
        )
    return {"documents": results}


@app.get("/documents/{filename}/preview")
def document_preview(filename: str):
    path = _validated_path(filename)

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        first_page = pdf.pages[0]
        page_text = (first_page.extract_text() or "").strip()[:PREVIEW_TEXT_CHARS]

    thumbnail_b64 = None
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(str(path), first_page=1, last_page=1, dpi=THUMBNAIL_DPI)
        if images:
            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            thumbnail_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        # Poppler may not be installed; degrade gracefully
        thumbnail_b64 = None

    return {
        "filename": filename,
        "page_count": page_count,
        "text_snippet": page_text,
        "thumbnail_png_base64": thumbnail_b64,
    }


@app.get("/documents/{filename}/chunks")
def document_chunks(filename: str):
    """Live view of exactly what's currently indexed for this document --
    reads straight from Chroma on every call, so it always reflects the
    real current state (no separate dump script or stale file to go stale)."""
    _validated_path(filename)

    def _fetch():
        collection = retrieval.get_collection()
        fetched = collection.get(where={"source": filename}, include=["documents", "metadatas"])
        docs = fetched.get("documents", [])
        metas = fetched.get("metadatas", [])
        ordered = sorted(
            zip(metas, docs), key=lambda pair: (pair[0].get("page", 0), pair[0].get("chunk_index", 0))
        )
        return [
            {
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "ocr_confidence": meta.get("ocr_confidence"),
                "text": text,
            }
            for meta, text in ordered
        ]

    chunks = retrieval.run_on_chroma_thread(_fetch)
    return {"filename": filename, "chunk_count": len(chunks), "chunks": chunks}


@app.get("/documents/{filename}/download")
def document_download(filename: str):
    path = _validated_path(filename)
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/pdf",
        content_disposition_type="inline",
    )


@app.post("/flag/{log_id}")
def flag(log_id: int, category: str, corrected_answer: Optional[str] = None):
    if category not in database.FLAG_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"category must be one of {database.FLAG_CATEGORIES}",
        )
    database.flag_answer(log_id, category, corrected_answer)
    return {"status": "flagged"}


@app.get("/export_flags")
def export_flags():
    return {"flags": database.export_flagged_answers()}


@app.post("/sessions")
def new_session():
    """Explicit session creation for the UI's "New chat" button -- distinct
    from /ask's implicit create-if-absent, so starting fresh is a real user
    action, not just "stop sending session_id and hope"."""
    return {"session_id": database.create_session()}


@app.get("/sessions")
def sessions():
    """Most-recent-first, each with an auto title and its running token
    total, so the UI can render a real, switchable session list instead of
    one invisible ongoing conversation."""
    return {"sessions": database.list_sessions()}


@app.get("/sessions/{session_id}/history")
def session_history(session_id: str):
    return {"messages": database.get_history(session_id)}


def _check_ollama() -> dict:
    start = time.perf_counter()
    try:
        resp = requests.get(config.OLLAMA_HOST, timeout=3)
        resp.raise_for_status()
        return {"online": True, "response_ms": round((time.perf_counter() - start) * 1000)}
    except Exception:
        return {"online": False, "response_ms": None}


@app.get("/monitor/status")
def monitor_status():
    return {
        "fastapi": {"online": True, "response_ms": 0},
        "ollama": _check_ollama(),
    }


_MONITOR_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DocRAG -- status</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 640px;
         margin: 3rem auto; padding: 0 1rem; background: #fff; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background: #17181c; color: #eaeaea; } }
  h1 { font-size: 20px; font-weight: 500; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .card { border: 1px solid rgba(128,128,128,0.3); border-radius: 12px; padding: 1rem 1.25rem; }
  .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .card-head span.name { font-weight: 500; font-size: 15px; }
  .status { display: flex; align-items: center; gap: 6px; font-size: 13px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .online { color: #1a9e56; } .online .dot { background: #1a9e56; }
  .offline { color: #d33; } .offline .dot { background: #d33; }
  table { width: 100%; font-size: 13px; border-collapse: collapse; }
  td { padding: 4px 0; }
  td.label { color: #888; }
  td.value { text-align: right; }
  .refresh-note { font-size: 13px; color: #888; margin-top: 12px; }
</style>
</head>
<body>
<h1>DocRAG -- system status</h1>
<div class="grid" id="grid"></div>
<p class="refresh-note">Auto-refreshes every 5s</p>
<script>
const SERVICES = [
  {key: "fastapi", name: "FastAPI backend", endpoint: "localhost:8000/health"},
  {key: "ollama", name: "Ollama (qwen2.5:3b)", endpoint: "127.0.0.1:11434"},
];

function render(status) {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const svc of SERVICES) {
    const s = status[svc.key] || {online: false, response_ms: null};
    const stateClass = s.online ? "online" : "offline";
    const stateLabel = s.online ? "Online" : "Offline";
    const responseText = s.response_ms === null || s.response_ms === undefined ? "-" : s.response_ms + " ms";
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-head">
        <span class="name">${svc.name}</span>
        <span class="status ${stateClass}"><span class="dot"></span>${stateLabel}</span>
      </div>
      <table>
        <tr><td class="label">Endpoint</td><td class="value">${svc.endpoint}</td></tr>
        <tr><td class="label">Response time</td><td class="value">${responseText}</td></tr>
        <tr><td class="label">Last checked</td><td class="value">just now</td></tr>
      </table>`;
    grid.appendChild(card);
  }
}

async function poll() {
  try {
    const resp = await fetch("/monitor/status");
    const status = await resp.json();
    render(status);
  } catch (e) {
    render({fastapi: {online: false, response_ms: null}, ollama: {online: false, response_ms: null}});
  }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>"""


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page():
    return _MONITOR_PAGE
