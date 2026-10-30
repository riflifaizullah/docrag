"""Builds questions_headings.json: one question per chosen section heading of the English documents.

    DOCRAG_DATA_DIR=<index folder> python benchmark/build_questions.py

The correct page is the page where the heading appears (or the next one, since a section often starts at the
bottom of a page). These questions share words with the heading, so they are the easy, lexical category.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "backend"))

from app import database, retrieval  # noqa: E402

PER_DOCUMENT = 10
NUMBER_PREFIX = re.compile(r"^\d+(\.\d+)*\.?\s+")
SKIP = re.compile(r"\.{4,}|^(table|fig|figure|appendix)\b|\b(figure|table)\s*\d|nist|sp 800|\d{4}|[.:;]$", re.I)


def clean(text: str) -> str | None:
    text = NUMBER_PREFIX.sub("", text.strip())
    words = text.split()
    letters = sum(c.isalpha() or c == " " for c in text)
    if SKIP.search(text) or not 3 <= len(words) <= 10 or letters / max(len(text), 1) < 0.9:
        return None
    return text


def main() -> None:
    english = {d["file"] for d in json.loads((HERE / "documents.json").read_text(encoding="utf-8")) if d["language"] == "en"}
    headings = retrieval.get_headings_for(sorted(database.list_document_sources()))
    counts: dict[str, int] = {}
    for h in headings:
        key = (clean(h["text"]) or "").lower()
        counts[key] = counts.get(key, 0) + 1

    questions = []
    for source in sorted(english):
        rows, seen = [], set()
        for h in sorted((h for h in headings if h["source"] == source and h["page"] >= 6), key=lambda h: (h["page"], h["line_no"])):
            text = clean(h["text"])
            if text and counts[text.lower()] == 1 and text.lower() not in seen:
                seen.add(text.lower())
                rows.append((text, h["page"]))
        step = max(1, len(rows) // PER_DOCUMENT)
        for text, page in rows[::step][:PER_DOCUMENT]:
            questions.append({
                "id": f"h{len(questions) + 1:03d}",
                "question": f"What does the document say about {text}?",
                "document": source,
                "pages": [page, page + 1],
            })
    (HERE / "questions_headings.json").write_text(json.dumps(questions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(questions), "questions")


if __name__ == "__main__":
    main()
