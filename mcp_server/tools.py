"""The tools the DocRAG MCP server offers. Each one calls the DocRAG backend over HTTP.

Kept free of the MCP library so it can be tested with only `requests`.
"""
import os

import requests

BACKEND_URL = os.getenv("DOCRAG_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# A full answer from a local model can take a while, so the timeout is long.
ASK_TIMEOUT_SECONDS = 600
QUICK_TIMEOUT_SECONDS = 60


class BackendError(RuntimeError):
    """Raised with a message that is safe and useful to show the person using the tool."""


def _call(method: str, path: str, timeout: int = QUICK_TIMEOUT_SECONDS, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise BackendError(
            f"The DocRAG backend is not reachable at {BACKEND_URL}. Start it first "
            "(docker compose up, or run the backend), or set DOCRAG_BACKEND_URL."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise BackendError(f"The DocRAG backend did not answer within {timeout} seconds.") from exc
    if response.status_code == 422:
        raise BackendError("The backend rejected the request, for example a question that is empty or longer than 4000 characters.")
    if not response.ok:
        raise BackendError(f"The DocRAG backend returned an error (HTTP {response.status_code}).")
    return response.json()


def list_documents() -> list[dict]:
    """The indexed documents: file name, type, page count and size in KB."""
    return _call("GET", "/documents")["documents"]


def search_passages(question: str, limit: int = 5) -> dict:
    """The best matching passages with their document and page. No language model is used."""
    return _call("POST", "/search", json={"question": question, "limit": max(1, min(limit, 20))})


def ask_documents(question: str) -> dict:
    """A written answer grounded in the documents, with the sources it used. Uses the local language model."""
    data = _call("POST", "/ask", timeout=ASK_TIMEOUT_SECONDS, json={"question": question})
    return {
        "answers": [
            {
                "question": r["question"],
                "answer": r["answer"],
                "sources": r["sources"],
                "refused": r["refused"],
                "needs_clarification": r["needs_clarification"],
                "candidate_documents": r["candidate_documents"],
            }
            for r in data["results"]
        ]
    }


def reindex_documents() -> dict:
    """Reads the documents folder again. New and changed files are indexed and removed files are dropped."""
    return _call("POST", "/sync", timeout=ASK_TIMEOUT_SECONDS)
