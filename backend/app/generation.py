"""Prompt construction, LLM calls, identifier grounding, fabrication safety
nets, multi-question splitting, and the ClarificationNeeded response shape.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import requests

from app import config, database, lang, retrieval

@dataclass
class ClarificationNeeded:
    message: str
    candidates: list[str] = field(default_factory=list)
    log_id: Optional[int] = None


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    log_id: Optional[int] = None
    candidates: list[str] = field(default_factory=list)


def _build_footer(notes: list[str]) -> str:
    """Caveats and the disclaimer, visually separated from the answer by a markdown divider and
    kept short.
    """
    return "\n\n---\n" + "\n".join(notes + [lang.get().SHORT_DISCLAIMER])


def build_prompt(question: str, hits: list[dict]) -> str:
    L = lang.get()
    # One document: name it once and label chunks by page only -- repeating the
    # full filename in every chunk label made the model echo it in every paragraph.
    sources = {h["source"] for h in hits}
    single_source = len(sources) == 1
    context_blocks = []
    for h in hits:
        label = (
            f"[{L.PAGE_WORD} {h['page']}]" if single_source
            else f"[{L.SOURCE_WORD}: {h['source']}, {L.PAGE_WORD} {h['page']}]"
        )
        context_blocks.append(f"{label}\n{h['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else L.NO_CONTEXT
    if single_source:
        context = f"{L.DOCUMENT_LABEL}: {next(iter(sources))}\n\n{context}"
    return (
        f"{L.SYSTEM_PROMPT}\n\n"
        f"{L.CONTEXT_LABEL}:\n{context}\n\n"
        f"{L.QUESTION_LABEL}: {question}\n\n"
        f"{L.ANSWER_LABEL}:"
    )


def build_full_document_prompt(question: str, source: str, full_text: str) -> str:
    """Used by the full-document fallback: the whole document in one block instead of retrieved
    chunks, for a broad/enumerate-style question about one small named document.
    """
    L = lang.get()
    return (
        f"{L.SYSTEM_PROMPT}\n\n"
        f"{L.FULL_DOCUMENT_LABEL} ({source}):\n{full_text}\n\n"
        f"{L.QUESTION_LABEL}: {question}\n\n"
        f"{L.ANSWER_LABEL}:"
    )


def _fits_in_context(text: str) -> bool:
    budget_tokens = config.OLLAMA_CONTEXT_TOKENS - config.FULL_DOC_PROMPT_MARGIN_TOKENS
    estimated_tokens = len(text) // config.CHARS_PER_TOKEN_ESTIMATE
    return estimated_tokens <= budget_tokens


def call_ollama(
    prompt: str, _retried: bool = False, timeout: Optional[int] = None
) -> tuple[str, Optional[int], Optional[int]]:
    """Returns (response_text, prompt_tokens, response_tokens)."""
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"num_ctx": config.OLLAMA_CONTEXT_TOKENS},
            },
            timeout=timeout or config.OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.Timeout):
        if _retried:
            raise
        return call_ollama(prompt, _retried=True, timeout=timeout)
    payload = resp.json()
    text = payload.get("response", "").strip()
    if not text and not _retried:
        return call_ollama(prompt, _retried=True, timeout=timeout)
    return (text, payload.get("prompt_eval_count"), payload.get("eval_count"))


_IDENTIFIER_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:[ \t-]+[A-Z][a-zA-Z]*)*)[ \t]+([A-Za-z]?\d+(?:[.\-]\d+)?)\b"
)


_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z]?\d+(?:[.\-]\d+)?$")


def heading_lead_words(headings: list[dict]) -> set[str]:
    """Vocabulary built dynamically per-call from the retrieved sources' real headings -- never
    a fixed hardcoded vocabulary.
    """
    vocab = set()
    for h in headings:
        text = h["text"].strip()
        for match in _IDENTIFIER_PATTERN.finditer(text):
            vocab.add(match.group(2).lower())
        if _BARE_IDENTIFIER_RE.match(text):
            vocab.add(text.lower())
    return vocab


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_FOOTER_SPLIT_RE = re.compile(r"\n\n---\n")


def _answer_sentences(answer_text: str) -> list[str]:
    """The answer's own claims, split into sentences -- excludes the
    footer this module appends (_build_footer: caveats + disclaimer), which
    is app-generated boilerplate, not a model claim to score."""
    body = _FOOTER_SPLIT_RE.split(answer_text or "", maxsplit=1)[0]
    sentences: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in _SENTENCE_SPLIT_RE.split(line):
            s = s.strip()
            if len(s.split()) >= 3:  # skip stray fragments/labels
                sentences.append(s)
    return sentences


def sentence_groundedness(answer_text: str, hits: list[dict]) -> dict:
    """Per-sentence semantic groundedness of `answer_text` against `hits`. Never raises."""
    try:
        sentences = _answer_sentences(answer_text)
        if not sentences or not hits:
            return {"min": None, "per_sentence": []}

        import numpy as np

        embedder = retrieval.get_embedder()
        sent_emb = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        chunk_emb = embedder.encode(
            [h.get("text", "") for h in hits], normalize_embeddings=True, show_progress_bar=False
        )
        sims = np.asarray(sent_emb) @ np.asarray(chunk_emb).T

        per_sentence = [
            {"sentence": s, "score": round(float(sims[i].max()), 4)} for i, s in enumerate(sentences)
        ]
        return {"min": min(p["score"] for p in per_sentence), "per_sentence": per_sentence}
    except Exception:
        traceback.print_exc()
        return {"min": None, "per_sentence": []}


def unverified_identifiers(
    answer_text: str, headings: list[dict], retrieved_pages: Optional[set[int]] = None
) -> list[str]:
    L = lang.get()
    vocab = heading_lead_words(headings)
    unverified = []
    for match in _IDENTIFIER_PATTERN.finditer(answer_text):
        if L.VALUE_UNIT_RE.match(answer_text[match.end():match.end() + 20]):
            continue
        if match.group(1).strip().lower() in L.MONTH_NAMES:
            continue
        if re.split(r"[ \t-]+", match.group(1).strip())[-1].lower() == L.PAGE_WORD.lower():
            # A page citation is not a heading number: check it against the
            # pages actually retrieved (skip when unknown, e.g. full-doc path).
            if retrieved_pages is None or match.group(2) in {str(p) for p in retrieved_pages}:
                continue
        elif match.group(2).lower() in vocab:
            continue
        if match.group(0) not in unverified:
            unverified.append(match.group(0))
    return unverified


def split_questions(text: str) -> list[str]:
    """Splits a combined message into sub-questions."""
    text = text.strip()
    if not text:
        return []

    L = lang.get()
    # strip trailing meta-instructions but keep them noted (not answered as Q)
    for pattern in L.META_REQUEST_PATTERNS:
        text = pattern.sub("", text).strip()

    raw_parts = re.split(
        rf"(?<=[?])\s+(?:{L.AND_WORDS})\s+|(?<=[?])\s+|\s+(?:{L.AND_WORDS})\s+(?={L.QUESTION_WORDS}\b)",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in raw_parts if p.strip()]

    if not parts:
        parts = [text]

    merged: list[str] = []
    for part in parts:
        if merged and L.REFERENCE_RE.search(part):
            joiner = " " if merged[-1].endswith("?") else f" {L.AND_JOINER} "
            merged[-1] = f"{merged[-1]}{joiner}{part}"
        else:
            merged.append(part)
    parts = merged

    final: list[str] = []
    for p in parts:
        if p and not re.fullmatch(r"[.!?,\s]*", p):
            final.append(p)
    return final


def is_followup_only(question: str) -> bool:
    return any(p.match(question) for p in lang.get().FOLLOWUP_PATTERNS)


_refusal_template_embeddings: dict = {}  # per language, built lazily to avoid loading the embedder at import
_SEMANTIC_REFUSAL_THRESHOLD = 0.55


def _looks_like_refusal_semantic(answer: str) -> bool:
    import numpy as np

    L = lang.get()
    embedder = retrieval.get_embedder()
    if L.CODE not in _refusal_template_embeddings:
        _refusal_template_embeddings[L.CODE] = embedder.encode(L.REFUSAL_TEMPLATES, normalize_embeddings=True)
    answer_emb = embedder.encode([answer[:200]], normalize_embeddings=True)[0]
    return float(np.max(_refusal_template_embeddings[L.CODE] @ answer_emb)) >= _SEMANTIC_REFUSAL_THRESHOLD


def looks_like_refusal(answer: str) -> bool:
    lower = answer.lower()
    if not any(phrase in lower for phrase in lang.get().REFUSAL_PHRASES):
        return False
    return _looks_like_refusal_semantic(answer)


def looks_like_hedging(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in lang.get().HEDGING_PHRASES)


def _passes_relevance_gate(hits: list[dict], question: Optional[str] = None) -> bool:
    if not hits:
        return False
    max_raw = max(h["score"] for h in hits)
    rerank_scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
    max_rerank = max(rerank_scores) if rerank_scores else float("-inf")
    if max_raw < config.RELEVANCE_MIN_SCORE and max_rerank < config.RELEVANCE_MIN_RERANK_SCORE:
        return False
    # A topic word found nowhere in the documents ("Kerberos") means the subject is probably not covered,
    # however similar the nearest passages look. Only a very strong match overrides that.
    if question and rerank_scores and max_rerank < config.UNKNOWN_TERM_RERANK_MIN and retrieval.unknown_terms(question):
        return False
    return True


def generate_answer(
    question: str, session_id: Optional[str] = None, _forced_source: Optional[str] = None
) -> Union[Answer, ClarificationNeeded]:
    L = lang.get()
    named_sources = [_forced_source] if _forced_source else retrieval.named_target_sources(question)

    if not named_sources:
        wide_hits = retrieval.retrieve(question, source_filter=None, k=config.MULTI_SOURCE_CANDIDATE_K)

        ambiguity = retrieval.detect_ambiguity(question, named_sources)
        if ambiguity is not None:
            message = L.CLARIFY.format(candidates=", ".join(ambiguity.candidates))
            log_id = database.log_answer(session_id, question, message, None, False, False, True, wide_hits)
            return ClarificationNeeded(message=message, candidates=ambiguity.candidates, log_id=log_id)

        if not _passes_relevance_gate(wide_hits, question):
            answer = L.NOT_FOUND
            log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
            return Answer(text=answer, sources=[], log_id=log_id)

        def _relevance(h: dict) -> float:
            return h.get("rerank_score", h["score"])

        best_per_source: dict[str, dict] = {}
        for h in wide_hits:
            src = h["source"]
            if src not in best_per_source or _relevance(h) > _relevance(best_per_source[src]):
                best_per_source[src] = h
        ranked_sources = sorted(best_per_source.items(), key=lambda kv: _relevance(kv[1]), reverse=True)

        if ranked_sources:
            top_score = _relevance(ranked_sources[0][1])
            ranked_sources = [
                (src, hit) for src, hit in ranked_sources
                if top_score - _relevance(hit) <= config.MULTI_SOURCE_RERANK_GAP
            ]

        if not ranked_sources:
            answer = L.NOT_FOUND
            log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
            return Answer(text=answer, sources=[], log_id=log_id)

        if len(ranked_sources) == 1:
            return generate_answer(question, session_id=session_id, _forced_source=ranked_sources[0][0])

        top_src = ranked_sources[0][0]
        top_answer = generate_answer(question, session_id=session_id, _forced_source=top_src)
        alternates = ranked_sources[1 : config.MULTI_SOURCE_CANDIDATE_LIST_MAX + 1]
        note = L.MULTI_SOURCE_NOTE.format(count=len(ranked_sources) - 1, top=top_src)
        all_candidates = [top_src] + [src for src, _ in alternates]
        combined_text = top_answer.text + note
        if top_answer.log_id is not None:
            database.update_answer_text(top_answer.log_id, combined_text, session_id)
        return Answer(
            text=combined_text,
            sources=top_answer.sources,
            log_id=top_answer.log_id,
            candidates=all_candidates,
        )

    broad = retrieval.broad_category_listing(question, named_sources)
    if broad is not None:
        lines = "\n".join(f"- {h}" for h in broad["headings"])
        answer = L.BROAD_LISTING.format(count=len(broad["headings"]), category=broad["category"], lines=lines)
        log_id = database.log_answer(session_id, question, answer, None, False, False, False, [])
        return Answer(text=answer, sources=broad["sources"], log_id=log_id)

    doc_listing = retrieval.document_type_listing(question, named_sources)
    if doc_listing is not None:
        lines = "\n".join(f"- {s}" for s in doc_listing["sources"])
        answer = L.DOC_TYPE_LISTING.format(
            count=len(doc_listing["sources"]), doc_type=doc_listing["doc_type"], lines=lines
        )
        log_id = database.log_answer(session_id, question, answer, None, False, False, False, [])
        return Answer(text=answer, sources=doc_listing["sources"], log_id=log_id)

    if len(named_sources) == 1 and not _IDENTIFIER_PATTERN.search(question):
        full_text = retrieval.get_full_document_text(named_sources[0])
        if (
            full_text
            and len(full_text) <= config.FULL_DOC_FALLBACK_MAX_CHARS
            and _fits_in_context(full_text)
        ):
            prompt = build_full_document_prompt(question, named_sources[0], full_text)
            answer, prompt_tokens, response_tokens = call_ollama(
                prompt, timeout=config.OLLAMA_FULL_DOC_TIMEOUT_SECONDS
            )
            headings = retrieval.get_headings_for(named_sources)
            unverified = unverified_identifiers(answer, headings)
            footer_notes = [L.FULL_DOC_NOTE]
            if unverified:
                footer_notes.append(L.UNVERIFIED_NOTE.format(items=", ".join(unverified)))
            answer += _build_footer(footer_notes)
            log_id = database.log_answer(
                session_id, question, answer, None, looks_like_hedging(answer),
                looks_like_refusal(answer), False, [],
                prompt_tokens=prompt_tokens, response_tokens=response_tokens,
            )
            return Answer(text=answer, sources=[named_sources[0]], log_id=log_id)

    source_filter = named_sources or None
    hits = retrieval.retrieve(question, source_filter=source_filter, k=config.TOP_K)

    if not hits:
        answer = L.NOT_FOUND
        log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
        return Answer(text=answer, sources=named_sources, log_id=log_id)

    prompt = build_prompt(question, hits)
    answer, prompt_tokens, response_tokens = call_ollama(prompt)

    relevant_sources = list({h["source"] for h in hits})
    headings = retrieval.get_headings_for(relevant_sources)
    unverified = unverified_identifiers(answer, headings, {h["page"] for h in hits})
    footer_notes = []
    if unverified:
        footer_notes.append(L.UNVERIFIED_NOTE.format(items=", ".join(unverified)))

    avg_score = sum(h["score"] for h in hits) / len(hits)

    ocr_confidences = [h["ocr_confidence"] for h in hits if h.get("ocr_confidence") is not None]
    avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
    if avg_ocr_confidence is not None and avg_ocr_confidence < database.LOW_OCR_CONFIDENCE_THRESHOLD:
        footer_notes.append(L.LOW_OCR_NOTE.format(confidence=avg_ocr_confidence))

    groundedness = sentence_groundedness(answer, hits)

    answer += _build_footer(footer_notes)
    refused = looks_like_refusal(answer)
    hedged = looks_like_hedging(answer)

    log_id = database.log_answer(
        session_id, question, answer, avg_score, hedged, refused, False, hits,
        avg_ocr_confidence=avg_ocr_confidence,
        prompt_tokens=prompt_tokens, response_tokens=response_tokens,
        min_groundedness=groundedness["min"], sentence_groundedness=groundedness["per_sentence"],
    )

    return Answer(text=answer, sources=relevant_sources, log_id=log_id)


def generate_multi_answer(text: str, session_id: Optional[str] = None) -> list[dict]:
    """Runs each sub-question through generate_answer()."""
    with lang.use(lang.detect(text)):
        return _generate_multi_answer(text, session_id)


def _generate_multi_answer(text: str, session_id: Optional[str]) -> list[dict]:
    L = lang.get()
    stripped = text.strip()
    resume_match = next((m for m in (mod.RESUME_RE.match(stripped) for mod in lang.all_modules()) if m), None)
    if resume_match and resume_match.group(2).strip() in retrieval.list_indexed_sources():
        sub_questions = [(resume_match.group(1).strip(), resume_match.group(2).strip())]
    else:
        sub_questions = [(q, None) for q in split_questions(text)]

    results = []
    for q, forced_source in sub_questions:
        try:
            result = generate_answer(q, session_id=session_id, _forced_source=forced_source)
        except Exception as exc:  # never drop a sub-question, surface the failure
            traceback.print_exc()  # full trace to stderr; user sees only the short message below
            results.append(
                {
                    "question": q,
                    "answer": L.PROCESSING_ERROR,
                    "answered": False,
                    "refused": False,
                    "needs_clarification": False,
                    "candidate_documents": [],
                    "sources": [],
                    "log_id": None,
                }
            )
            continue

        if isinstance(result, ClarificationNeeded):
            results.append(
                {
                    "question": q,
                    "answer": result.message,
                    "answered": False,
                    "refused": False,
                    "needs_clarification": True,
                    "candidate_documents": result.candidates,
                    "sources": [],
                    "log_id": result.log_id,
                }
            )
        else:
            refused = looks_like_refusal(result.text)
            results.append(
                {
                    "question": q,
                    "answer": result.text,
                    "answered": not refused,
                    "refused": refused,
                    "needs_clarification": False,
                    "candidate_documents": result.candidates,
                    "sources": result.sources,
                    "log_id": result.log_id,
                }
            )
    return results
