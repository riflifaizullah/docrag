"""SQLite persistence: chat/session history, headings index, document sync
registry, and the answer-flagging log.

Kept as one small module per the "share reconciliation logic" and "avoid
duplicated storage" guidance in the rebuild spec -- everything that touches
SQLite goes through here.
"""
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

from app import config

_local = threading.local()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                file_fingerprint TEXT,
                page_count INTEGER,
                size_kb REAL
            );

            CREATE TABLE IF NOT EXISTS headings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                text TEXT NOT NULL,
                line_no INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_headings_source ON headings(source);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

            CREATE TABLE IF NOT EXISTS answer_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                retrieval_confidence REAL,
                hedging_detected INTEGER NOT NULL DEFAULT 0,
                refused INTEGER NOT NULL DEFAULT 0,
                needs_clarification INTEGER NOT NULL DEFAULT 0,
                retrieved_context TEXT,
                created_at TEXT NOT NULL,
                human_flag INTEGER NOT NULL DEFAULT 0,
                flag_category TEXT,
                corrected_answer TEXT,
                avg_ocr_confidence REAL,
                low_ocr_confidence INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER,
                response_tokens INTEGER,
                min_groundedness REAL,
                sentence_groundedness TEXT
            );

            CREATE TABLE IF NOT EXISTS vlm_cache (
                image_hash TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        # migration for DBs created before the OCR-confidence/token columns existed
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(answer_log)").fetchall()}
        if "avg_ocr_confidence" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN avg_ocr_confidence REAL")
        if "low_ocr_confidence" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN low_ocr_confidence INTEGER NOT NULL DEFAULT 0")
        if "prompt_tokens" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN prompt_tokens INTEGER")
        if "response_tokens" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN response_tokens INTEGER")
        if "min_groundedness" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN min_groundedness REAL")
        if "sentence_groundedness" not in existing_cols:
            conn.execute("ALTER TABLE answer_log ADD COLUMN sentence_groundedness TEXT")

        # migration for DBs created before file_fingerprint existed
        existing_doc_cols = {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "file_fingerprint" not in existing_doc_cols:
            conn.execute("ALTER TABLE documents ADD COLUMN file_fingerprint TEXT")
        if "page_count" not in existing_doc_cols:
            conn.execute("ALTER TABLE documents ADD COLUMN page_count INTEGER")
        if "size_kb" not in existing_doc_cols:
            conn.execute("ALTER TABLE documents ADD COLUMN size_kb REAL")


# ---------------------------------------------------------------------------
# Document sync registry
# ---------------------------------------------------------------------------

def get_document_hash(source: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE source = ?", (source,)
        ).fetchone()
        return row["content_hash"] if row else None


def get_document_fingerprint(source: str) -> Optional[str]:
    """Cheap file-metadata fingerprint (mtime + size), checked before paying
    for a full content_hash() read -- see sync_documents.sync_once(). At
    thousands-of-files scale, hashing every file's full bytes on every sync
    tick just to confirm "unchanged" is real, avoidable I/O; mtime+size
    matching is the same fast-path trick git/rsync use before falling back
    to a real hash."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT file_fingerprint FROM documents WHERE source = ?", (source,)
        ).fetchone()
        return row["file_fingerprint"] if row else None


def upsert_document(
    source: str,
    content_hash: str,
    file_fingerprint: Optional[str] = None,
    page_count: Optional[int] = None,
    size_kb: Optional[float] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO documents (source, content_hash, indexed_at, file_fingerprint, page_count, size_kb)
            VALUES (?, ?, datetime('now'), ?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                content_hash = excluded.content_hash,
                indexed_at = excluded.indexed_at,
                file_fingerprint = excluded.file_fingerprint,
                page_count = excluded.page_count,
                size_kb = excluded.size_kb
            """,
            (source, content_hash, file_fingerprint, page_count, size_kb),
        )


def delete_document(source: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE source = ?", (source,))
        conn.execute("DELETE FROM headings WHERE source = ?", (source,))


def get_document_metadata(sources: list[str]) -> dict[str, tuple[Optional[int], Optional[float]]]:
    """Bulk (page_count, size_kb) lookup for /documents -- avoids opening
    every PDF with pdfplumber on every request (confirmed live: doing that
    for a thousand-plus files took 40+ seconds, past the UI's fetch timeout, which is
    why the sidebar showed 0 documents). Missing/never-backfilled sources
    are simply absent from the returned dict."""
    if not sources:
        return {}
    with get_conn() as conn:
        placeholders = ",".join("?" * len(sources))
        rows = conn.execute(
            f"SELECT source, page_count, size_kb FROM documents WHERE source IN ({placeholders})",
            sources,
        ).fetchall()
    return {r["source"]: (r["page_count"], r["size_kb"]) for r in rows if r["page_count"] is not None}


def set_document_metadata(source: str, page_count: int, size_kb: float) -> None:
    """Backfill path for documents indexed before page_count/size_kb existed
    -- see get_document_metadata()."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET page_count = ?, size_kb = ? WHERE source = ?",
            (page_count, size_kb, source),
        )


def get_vlm_description(image_hash: str) -> Optional[str]:
    """The VLM's own sampling isn't fully deterministic even at
    temperature=0 (confirmed live -- likely GPU floating-point execution
    order, not fixable at the request level), so re-ingesting an unchanged
    file could get a different diagram description and therefore a
    different chunk count each time. The same rendered page image always
    hashes the same, though -- caching by that hash makes RE-ingestion of
    unchanged content deterministic without needing the model itself to
    be reproducible."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT description FROM vlm_cache WHERE image_hash = ?", (image_hash,)
        ).fetchone()
        return row["description"] if row else None


def set_vlm_description(image_hash: str, description: str) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO vlm_cache (image_hash, description, created_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(image_hash) DO UPDATE SET description = excluded.description
            """,
            (image_hash, description),
        )


def list_document_sources() -> list[str]:
    """Paginated read to avoid hitting SQL-variable limits at large corpus size."""
    sources: list[str] = []
    with get_conn() as conn:
        offset = 0
        while True:
            rows = conn.execute(
                "SELECT source FROM documents ORDER BY source LIMIT ? OFFSET ?",
                (config.CHROMA_PAGE_SIZE, offset),
            ).fetchall()
            if not rows:
                break
            sources.extend(r["source"] for r in rows)
            offset += config.CHROMA_PAGE_SIZE
    return sources


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def replace_headings(source: str, headings: Iterable[dict]) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM headings WHERE source = ?", (source,))
        conn.executemany(
            "INSERT INTO headings (source, page, text, line_no) VALUES (?, ?, ?, ?)",
            [(source, h["page"], h["text"], h["line_no"]) for h in headings],
        )


def get_headings_for_sources(sources: Iterable[str]) -> list[dict]:
    """Heading lookups MUST be scoped to specific document(s) -- never the
    whole corpus. See known-bug #2 (cross-document heading contamination)."""
    sources = list(sources)
    if not sources:
        return []
    results: list[dict] = []
    with get_conn() as conn:
        # paginate the IN clause in batches to respect SQLite's variable limit
        batch_size = 200
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT source, page, text, line_no FROM headings "
                f"WHERE source IN ({placeholders}) ORDER BY source, page, line_no",
                batch,
            ).fetchall()
            results.extend(dict(r) for r in rows)
    return results


def get_all_headings_grouped_by_source() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for h in get_headings_for_sources(list_document_sources()):
        grouped.setdefault(h["source"], []).append(h)
    return grouped


# ---------------------------------------------------------------------------
# Chat / sessions
# ---------------------------------------------------------------------------

def create_session() -> str:
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, datetime('now'))",
            (session_id,),
        )
    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (session_id, role, content),
        )


def get_history(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_sessions() -> list[dict]:
    """Most-recent-first, each with an auto title (its first user message,
    truncated) and a running token total -- reads straight from the
    sessions/messages/answer_log tables that already existed but were never
    surfaced anywhere; no new persistence concept, just exposing it."""
    with get_conn() as conn:
        sessions = conn.execute(
            "SELECT session_id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for s in sessions:
            sid = s["session_id"]
            first_q = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id ASC LIMIT 1",
                (sid,),
            ).fetchone()
            title = (first_q["content"][:60] + "...") if first_q and len(first_q["content"]) > 60 else (
                first_q["content"] if first_q else "(sesi kosong)"
            )
            totals = conn.execute(
                "SELECT COALESCE(SUM(prompt_tokens), 0) AS p, COALESCE(SUM(response_tokens), 0) AS r "
                "FROM answer_log WHERE session_id = ?",
                (sid,),
            ).fetchone()
            result.append(
                {
                    "session_id": sid,
                    "created_at": s["created_at"],
                    "title": title,
                    "prompt_tokens": totals["p"],
                    "response_tokens": totals["r"],
                    "total_tokens": totals["p"] + totals["r"],
                }
            )
        return result


# ---------------------------------------------------------------------------
# Answer flagging (proxy signals now, human review later)
# ---------------------------------------------------------------------------

LOW_OCR_CONFIDENCE_THRESHOLD = 60.0  # Tesseract mean word confidence is 0-100

# Fixed vocabulary for human flags, not free-form text -- keeps the flagged
# set analyzable later (group by category) instead of a pile of one-off
# strings. Category is free-form in storage (no CHECK constraint), but every
# writer (flag_answer's caller in app/api.py) is expected to send one of these.
FLAG_CATEGORIES: tuple[str, ...] = (
    "hallucinated_fact",
    "wrongly_refused",
    "bad_retrieval",
    "incomplete_or_unclear",
    "other",
)


def log_answer(
    session_id: Optional[str],
    question: str,
    answer: str,
    retrieval_confidence: Optional[float],
    hedging_detected: bool,
    refused: bool,
    needs_clarification: bool,
    retrieved_context: list[dict],
    avg_ocr_confidence: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    response_tokens: Optional[int] = None,
    min_groundedness: Optional[float] = None,
    sentence_groundedness: Optional[list[dict]] = None,
) -> int:
    low_ocr_confidence = avg_ocr_confidence is not None and avg_ocr_confidence < LOW_OCR_CONFIDENCE_THRESHOLD
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO answer_log (
                session_id, question, answer, retrieval_confidence,
                hedging_detected, refused, needs_clarification,
                retrieved_context, created_at, avg_ocr_confidence, low_ocr_confidence,
                prompt_tokens, response_tokens, min_groundedness, sentence_groundedness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                retrieval_confidence,
                int(hedging_detected),
                int(refused),
                int(needs_clarification),
                json.dumps(retrieved_context, ensure_ascii=False),
                avg_ocr_confidence,
                int(low_ocr_confidence),
                prompt_tokens,
                response_tokens,
                min_groundedness,
                json.dumps(sentence_groundedness or [], ensure_ascii=False),
            ),
        )
        row_id = cur.lastrowid
    if session_id:
        _write_session_log_file(session_id)
    return row_id


def update_answer_text(log_id: int, answer: str, session_id: Optional[str] = None) -> None:
    """Overwrite an already-logged answer's text in place -- for the one
    case where the text a user actually sees is finalized AFTER the row
    was written (generation.py's competing-documents branch logs the inner,
    single-source answer first via its own recursive log_answer() call,
    then appends a "other documents also matched" note and candidate list
    on top of it). Without this, the persisted answer_log row -- and
    therefore flagging and any eval reading it -- would silently see
    different text than what the UI actually rendered (caught in code
    review, not live)."""
    with get_conn() as conn:
        conn.execute("UPDATE answer_log SET answer = ? WHERE id = ?", (answer, log_id))
    if session_id:
        _write_session_log_file(session_id)


def _write_session_log_file(session_id: str) -> None:
    """Human-readable per-session Markdown log under data/session_logs/,
    rewritten in full from answer_log on every turn -- SQLite stays the one
    source of truth (matches list_sessions()'s token totals exactly), the
    file is just a always-current, easy-to-open view of it. One block per
    answer_log row (i.e. per sub-question when a turn got split into
    several), since that's the granularity token counts actually exist at;
    a multi-question turn showing up as separate Q&A blocks is a feature,
    not a loss of information."""
    with get_conn() as conn:
        session = conn.execute(
            "SELECT created_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        rows = conn.execute(
            "SELECT question, answer, prompt_tokens, response_tokens, created_at "
            "FROM answer_log WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    total_prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    total_response = sum(r["response_tokens"] or 0 for r in rows)
    lines = [
        f"# Sesi {session_id}",
        f"Dibuat: {session['created_at'] if session else 'tidak diketahui'}",
        f"Total token: {total_prompt + total_response} (prompt: {total_prompt}, response: {total_response})",
        "",
    ]
    for r in rows:
        turn_tokens = (r["prompt_tokens"] or 0) + (r["response_tokens"] or 0)
        lines.append(f"## {r['created_at']} -- {turn_tokens} token")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        lines.append(f"**A:** {r['answer']}")
        lines.append("")

    log_path = config.SESSION_LOGS_DIR / f"{session_id}.md"
    log_path.write_text("\n".join(lines), encoding="utf-8")


def flag_answer(log_id: int, category: str, corrected_answer: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE answer_log SET human_flag = 1, flag_category = ?, corrected_answer = ? "
            "WHERE id = ?",
            (category, corrected_answer, log_id),
        )


def export_flagged_answers() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM answer_log WHERE human_flag = 1 ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Short-TTL cache helper (used by retrieval.list_indexed_sources)
# ---------------------------------------------------------------------------

class TTLCache:
    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._value = None
        self._set_at = 0.0

    def get_or_set(self, factory):
        now = time.monotonic()
        if self._value is None or (now - self._set_at) > self.ttl_seconds:
            self._value = factory()
            self._set_at = now
        return self._value

    def invalidate(self):
        self._value = None
        self._set_at = 0.0
