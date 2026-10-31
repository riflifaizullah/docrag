"""Demo mode (DOCRAG_DEMO=1): a read-only DocRAG that is safe to put on a public address.

No language model is called. A question returns the best matching passages with their documents and pages, or the
usual "not found" when nothing relevant was retrieved. Changing endpoints are switched off and questions are rate
limited. The backend still has no sign-in, so demo mode is for public sample documents only.
"""
import threading
import time
from typing import Optional

from fastapi import HTTPException

from app import config, generation, lang, retrieval

PASSAGES_SHOWN = 3
PASSAGE_CHARS = 450


class RateLimiter:
    """At most `max_calls` in any `per_seconds` window, shared by all visitors."""

    def __init__(self, max_calls: int, per_seconds: float, clock=time.monotonic):
        self.max_calls, self.per_seconds, self._clock = max_calls, per_seconds, clock
        self._calls: list[float] = []
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = self._clock()
        with self._lock:
            self._calls = [t for t in self._calls if now - t < self.per_seconds]
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


limiter = RateLimiter(config.DEMO_QUESTIONS_PER_MINUTE, 60)


def require_writable() -> None:
    """Dependency for endpoints that change data or reveal internals."""
    if config.DEMO_MODE:
        raise HTTPException(status_code=403, detail="This is a read-only demo.")


def check_rate_limit() -> None:
    if config.DEMO_MODE and not limiter.allow():
        raise HTTPException(status_code=429, detail="The demo is busy. Please try again in a minute.")


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= PASSAGE_CHARS else text[:PASSAGE_CHARS].rsplit(" ", 1)[0] + " ..."


def _close_to_best(hits: list[dict]) -> list[dict]:
    """Passages whose rerank score is within the usual gap of the best one, so weak matches are not listed."""
    scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
    if not scores:
        return hits
    floor = max(scores) - config.MULTI_SOURCE_RERANK_GAP
    return [h for h in hits if h.get("rerank_score", floor) >= floor]


def passages_answer(question: str, hits: Optional[list[dict]] = None) -> dict:
    """The fields of one answer made only from retrieved passages."""
    with lang.use(lang.detect(question)):
        L = lang.get()
        hits = retrieval.retrieve(question, k=PASSAGES_SHOWN) if hits is None else hits
        if not generation.passes_relevance_gate(hits, question):
            return {"question": question, "answer": L.NOT_FOUND, "answered": False, "refused": True,
                    "needs_clarification": False, "candidate_documents": [], "sources": [], "log_id": None}
        shown = _close_to_best(hits)[:PASSAGES_SHOWN]
        lines = [f"- {_short(h['text'])} ({h['source']}, p. {h['page']})" for h in shown]
        sources = list(dict.fromkeys(h["source"] for h in shown))
        return {"question": question, "answer": L.DEMO_ANSWER_INTRO + "\n" + "\n".join(lines), "answered": True,
                "refused": False, "needs_clarification": False, "candidate_documents": [], "sources": sources,
                "log_id": None}
