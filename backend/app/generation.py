"""Prompt construction, LLM calls, identifier grounding, fabrication safety
nets, multi-question splitting, and the ClarificationNeeded response shape.
"""
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import requests

from app import config, database, retrieval

REFUSAL_PHRASES = [
    # Indonesian
    # Kept broad on purpose: a bare "tidak ditemukan" can false-positive on
    # legitimate domain content (a real clinical "Tidak ditemukan kelainan
    # medis" category). Narrowing the PHRASE itself broke more genuine
    # refusals than it fixed (tried "tidak ditemukan dalam dokumen": 6/8
    # real refusals phrase it differently) and a length cutoff also failed
    # (a genuine long, helpful refusal was as long as the false positive).
    # Real fix lives one level up, in looks_like_refusal(): a phrase match
    # here is only a candidate: semantic similarity against the system's
    # own actual refusal framing is what disambiguates content from a real
    # refusal, since this list can't.
    "tidak ditemukan", "tidak tersedia", "tidak dapat menemukan",
    "maaf, saya tidak", "tidak ada informasi", "di luar cakupan",
    "tidak disebutkan dalam dokumen",
    # English (known-bug #6: evaluator must recognize both languages)
    "cannot find", "not available in the document", "no information",
    "i cannot answer", "not mentioned in the document", "out of scope",
    "i don't have that information",
]

HEDGING_PHRASES = [
    "mungkin", "kemungkinan", "sepertinya", "perlu diverifikasi",
    "possibly", "it seems", "may be", "not certain", "unclear",
]


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
    # Other documents that also looked relevant when none was named in the
    # question -- the UI offers these as a "narrow it down" choice, same
    # candidate-button mechanism ClarificationNeeded uses, but alongside a
    # real answer instead of blocking on one (see generate_answer()).
    candidates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Anda adalah asisten tanya jawab untuk kumpulan dokumen.
Jawab HANYA berdasarkan konteks yang diberikan di bawah ini. Jangan mengarang informasi.
Jika jawabannya tidak ada dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak ditemukan dalam dokumen.
Jika beberapa sumber memberikan nilai yang berbeda untuk fakta yang tampaknya sama, sebutkan nilai dari masing-masing
sumber secara terpisah dengan menyebut nama dokumennya, jangan menggabungkannya menjadi satu jawaban yang kontradiktif.
Jangan menyalin label konteks seperti "[Sumber: ...]" ke dalam jawaban, dan jangan mengawali setiap paragraf dengan
frasa yang sama. Cukup rujuk halaman secara singkat di akhir kalimat, misalnya (hlm. 24).

Pilih format jawaban sesuai bentuk isinya, jangan dipaksakan satu bentuk untuk semua kasus:
- Jika jawabannya satu fakta sederhana, tulis sebagai satu paragraf singkat yang mengalir, seperti menjelaskan
  langsung ke rekan kerja.
- Jika isinya memang terdiri dari beberapa bagian yang berbeda (misalnya beberapa tahap/stage, beberapa
  langkah, atau beberapa sub-item dengan nilai masing-masing), tulis satu baris singkat per bagian dengan
  label tebal (**Label** -- penjelasan singkat), bukan dipaksakan jadi satu paragraf panjang.
Jangan menyalin ulang kalimat konteks kata per kata atau mengulang frasa templat yang sama untuk setiap
ayat/sub-bagian. Sebutkan angka, tanggal, dan kode referensi persis apa adanya (jangan diubah nilainya)."""


_SHORT_DISCLAIMER = "Dihasilkan otomatis -- verifikasi ulang untuk hal penting."


def _build_footer(notes: list[str]) -> str:
    """Caveats and the disclaimer, visually separated from the answer by a
    markdown divider and kept short -- previously appended as bracketed
    sentences stitched directly onto the answer's own prose, which read
    as clutter rather than a distinct, skimmable caveat."""
    return "\n\n---\n" + "\n".join(notes + [_SHORT_DISCLAIMER])


def build_prompt(question: str, hits: list[dict]) -> str:
    # One document: name it once and label chunks by page only -- repeating the
    # full filename in every chunk label made the model echo it in every paragraph.
    sources = {h["source"] for h in hits}
    single_source = len(sources) == 1
    context_blocks = []
    for h in hits:
        label = f"[Halaman {h['page']}]" if single_source else f"[Sumber: {h['source']}, Halaman {h['page']}]"
        context_blocks.append(f"{label}\n{h['text']}")
    context = "\n\n".join(context_blocks) if context_blocks else "(tidak ada konteks yang relevan)"
    if single_source:
        context = f"DOKUMEN: {next(iter(sources))}\n\n{context}"
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"KONTEKS:\n{context}\n\n"
        f"PERTANYAAN: {question}\n\n"
        f"JAWABAN:"
    )


def build_full_document_prompt(question: str, source: str, full_text: str) -> str:
    """Used by the full-document fallback: the whole document in one block
    instead of retrieved chunks, for a broad/enumerate-style question about
    one small named document -- guarantees completeness regardless of
    chunk-level retrieval quality, since there's no top-k to drop anything
    from."""
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"DOKUMEN LENGKAP ({source}):\n{full_text}\n\n"
        f"PERTANYAAN: {question}\n\n"
        f"JAWABAN:"
    )


def _fits_in_context(text: str) -> bool:
    budget_tokens = config.OLLAMA_CONTEXT_TOKENS - config.FULL_DOC_PROMPT_MARGIN_TOKENS
    estimated_tokens = len(text) // config.CHARS_PER_TOKEN_ESTIMATE
    return estimated_tokens <= budget_tokens


def call_ollama(
    prompt: str, _retried: bool = False, timeout: Optional[int] = None
) -> tuple[str, Optional[int], Optional[int]]:
    """Returns (response_text, prompt_tokens, response_tokens). Ollama's
    payload carries these as prompt_eval_count/eval_count on every call --
    previously read only for the response text and the counts discarded,
    so per-session usage was invisible without digging through Ollama's own
    logs. Token counts are the app's own runtime LLM usage.

    Explicit num_ctx matters here: without it, Ollama runs this model at
    its own default context, which can silently be smaller than
    OLLAMA_CONTEXT_TOKENS -- the value this app's own size-fit checks
    (_fits_in_context) assume is available. Passing it explicitly makes
    the real running context match what the app already believes it is,
    the same fix already applied to the VLM's context in ingestion.py."""
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                # qwen3.5 is a reasoning model -- confirmed live it burns
                # ~9000 chars / 2300+ tokens of hidden <thinking> content
                # per call by default (78s), thrown away before the visible
                # answer is even built. This RAG task is extraction/
                # summarization over already-retrieved passages, not
                # multi-step reasoning, so the hidden reasoning buys nothing
                # here. Disabling it: same final answer text, 1.7s instead
                # of 78s on the same question -- confirmed live, not assumed.
                "think": False,
                "options": {"num_ctx": config.OLLAMA_CONTEXT_TOKENS},
            },
            timeout=timeout or config.OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except (requests.exceptions.HTTPError, requests.exceptions.Timeout):
        # On this 8GB-RAM laptop, Ollama unloads the idle model and an
        # occasional first request after reload transiently 500s while it
        # reloads under memory pressure. Confirmed live this can also
        # surface as an outright Timeout rather than a clean HTTP error --
        # a large full-document-fallback prompt exceeded
        # OLLAMA_TIMEOUT_SECONDS right after another model (the VLM) had
        # been swapped into VRAM. Same transient cause, same one-retry fix;
        # a second failure is a real error and should propagate.
        if _retried:
            raise
        return call_ollama(prompt, _retried=True, timeout=timeout)
    payload = resp.json()
    text = payload.get("response", "").strip()
    if not text and not _retried:
        # Confirmed live: a request can come back with a real 200 status
        # but an empty response body -- observed once during heavy Ollama
        # load (another model being swapped into VRAM at the same time).
        # Not an HTTPError, so the retry above never caught it; one retry
        # clears it the same way a transient reload does.
        return call_ollama(prompt, _retried=True, timeout=timeout)
    return (text, payload.get("prompt_eval_count"), payload.get("eval_count"))


# ---------------------------------------------------------------------------
# Identifier grounding (known-bug #8: pattern must accept letters, not just
# digits -- and, per the audit that found this still hardcoded to
# Pasal/Lampiran/Bab, must not assume which keyword a document type uses
# for its identifiers. "Ayat" was already missing from this list, and any
# future document type coining its own identifier word would have been
# invisible to fabrication-checking. Mirrors retrieval.py's
# _IDENTIFIER_EXTRACT_RE: any capitalized word/phrase + alphanumeric token,
# no fixed vocabulary.)
# ---------------------------------------------------------------------------

# [ \t] (not \s) between phrase words and before the number: a real
# section identifier is always written on one line -- \s would also match
# a newline, so a numbered list the LLM writes in its own answer ("...
# Budi Santoso\n2. Perihal: ...") reads as one fabricated "phrase +
# number" pair spanning the list marker's line break, not an actual
# document reference (confirmed live). Same phrase used in retrieval.py's
# _IDENTIFIER_EXTRACT_RE should mirror this if it's ever seen doing the
# same thing against multi-line text.
_IDENTIFIER_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:[ \t-]+[A-Z][a-zA-Z]*)*)[ \t]+([A-Za-z]?\d+(?:[.\-]\d+)?)\b"
)


_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z]?\d+(?:[.\-]\d+)?$")


def heading_lead_words(headings: list[dict]) -> set[str]:
    """Vocabulary built dynamically per-call from the retrieved sources'
    real headings -- never a fixed hardcoded vocabulary. Keyed on the
    identifier's number/code alone, not the "word number" pair: a contract
    ayat marker is drawn as its own short bold line ("1.2") with no "Ayat"
    or "Pasal" prefix in the text detect_headings() actually stores, so
    matching on the full pair produced false positives -- a genuine "Ayat
    1.2" reference in an answer got flagged as unverified purely because
    the stored heading text never spells out which word it belongs to.
    The number is still specific enough within one document to catch real
    fabrications (e.g. "Pasal 99" when the highest real Pasal is 25)."""
    vocab = set()
    for h in headings:
        text = h["text"].strip()
        for match in _IDENTIFIER_PATTERN.finditer(text):
            vocab.add(match.group(2).lower())
        if _BARE_IDENTIFIER_RE.match(text):
            vocab.add(text.lower())
    return vocab


# A fully generic "CapitalizedWord + number" pattern also matches ordinary
# value phrases, not just section identifiers -- e.g. a bulleted answer
# like "- Ayat 21.4: Sebesar 74 juta rupiah" makes "Sebesar 74" look like a
# candidate identifier purely because "Sebesar" is capitalized as the first
# word after the colon. This isn't the same trap as hardcoding a corpus's
# document-type words (part 1) or section-keyword words (identifier
# vocabulary above): Indonesian/English measurement and currency units are
# a closed, stable part of the language itself, not something that varies
# by document type or corpus, so filtering on "the number is immediately
# followed by a unit word" is a linguistic fact, not a naming-convention
# guess that could go stale.
_VALUE_UNIT_RE = re.compile(
    r"^\s*(juta|ribu|miliar|persen|%|rupiah|hari|bulan|tahun|menit|jam|kali)\b",
    re.IGNORECASE,
)

# A calendar date ("1 November 2024", "September 2025") makes the generic
# "CapitalizedWord + number" pattern look like a section identifier purely
# because a month name is capitalized and immediately precedes a year --
# confirmed live: "berlaku mulai tanggal 1 November 2024" got flagged as an
# unverified identifier. Same category of fix as the value-unit filter
# above: Indonesian/English month names are a closed, stable part of the
# language, not corpus-specific vocabulary that could go stale.
_MONTH_NAMES = {
    "januari", "februari", "maret", "april", "mei", "juni", "juli", "agustus",
    "september", "oktober", "november", "desember",
    "january", "february", "march", "june", "july", "august", "october", "december",
}


# ---------------------------------------------------------------------------
# Sentence-level semantic groundedness (signal only, not enforced)
# ---------------------------------------------------------------------------
# unverified_identifiers() catches fabricated section/clause NUMBERS. It
# cannot catch a fluent answer that bridges two genuinely-retrieved but
# unrelated facts into a claim neither actually supports -- there is no
# identifier or keyword to regex for there, only semantic distance between
# what the answer claims and what the passages actually say. This mirrors
# the older RAG SYSTEM's sentence_groundedness() (same technique: reuse the
# retrieval embedder, no new dependency, no new model). Deliberately a
# SIGNAL only, logged per answer for later review -- not a gate, since
# nothing here has been validated as reliable enough to block an answer on.
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
    """Per-sentence semantic groundedness of `answer_text` against `hits` --
    for each answer sentence, its score is the MAX cosine similarity against
    any one retrieved passage (embeddings pre-normalized, so cosine
    similarity is a plain dot product). Returns {"min": float|None,
    "per_sentence": [{"sentence","score"}, ...]}; `min` is None when there
    is nothing to score (empty answer or no hits) -- callers must not treat
    None as "ungrounded", only as "not evaluated". Never raises."""
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
    vocab = heading_lead_words(headings)
    unverified = []
    for match in _IDENTIFIER_PATTERN.finditer(answer_text):
        if _VALUE_UNIT_RE.match(answer_text[match.end():match.end() + 20]):
            continue
        if match.group(1).strip().lower() in _MONTH_NAMES:
            continue
        if re.split(r"[ \t-]+", match.group(1).strip())[-1].lower() == "halaman":
            # A page citation is not a heading number: check it against the
            # pages actually retrieved (skip when unknown, e.g. full-doc path).
            if retrieved_pages is None or match.group(2) in {str(p) for p in retrieved_pages}:
                continue
        elif match.group(2).lower() in vocab:
            continue
        if match.group(0) not in unverified:
            unverified.append(match.group(0))
    return unverified


# ---------------------------------------------------------------------------
# Multi-question splitting
# ---------------------------------------------------------------------------

_META_REQUEST_PATTERNS = [
    # the leading connector ("tolong"/"please") is consumed together with
    # the instruction phrase -- stripping only the phrase and leaving the
    # connector behind created an orphan fragment ("Tolong .") that then
    # got treated as its own phantom sub-question (found via live testing
    # of "... dan apa isi Lampiran A ...? Tolong jawab singkat.").
    re.compile(
        r"\b(?:tolong|please)\s+(?:jawab\s+singkat|be\s+brief|be\s+detailed|detail(?:kan)?|jelaskan\s+secara\s+detail)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:jawab\s+singkat|be\s+brief|be\s+detailed|jelaskan\s+secara\s+detail)\b", re.IGNORECASE),
]

_FOLLOWUP_PATTERNS = [
    re.compile(r"^\s*(why\?|kenapa\?|mengapa\?)\s*$", re.IGNORECASE),
]


_QUESTION_WORD = (
    r"(?:apa|berapa|siapa|kapan|dimana|di\s+mana|bagaimana|mengapa|kenapa|"
    r"what|how|when|where|why|who|which)"
)


def split_questions(text: str) -> list[str]:
    """Splits a combined message into sub-questions. Handles conjunctions,
    trailing meta-instructions, and short context-dependent follow-ups.
    A sub-question must never be silently dropped."""
    text = text.strip()
    if not text:
        return []

    # strip trailing meta-instructions but keep them noted (not answered as Q)
    for pattern in _META_REQUEST_PATTERNS:
        text = pattern.sub("", text).strip()

    # Split on: a "?" followed by a conjunction, a bare "?", or a
    # conjunction immediately followed by a question word even with no
    # preceding "?" ("...di Kontrak_Sewa_100.pdf dan apa isi Lampiran A...").
    # That last case was a real gap found in testing: a compound question
    # with only one trailing "?" got sent through generate_answer() as one
    # combined question -- it happened to answer both facts because a
    # single retrieval call covered both, but nothing guaranteed that, and
    # a genuine drop wouldn't have been visible. The question-word lookahead
    # keeps this from over-splitting an ordinary single question like
    # "Pasal 2 dan Pasal 3" that just happens to contain "dan".
    raw_parts = re.split(
        rf"(?<=[?])\s+(?:dan|and)\s+|(?<=[?])\s+|\s+(?:dan|and)\s+(?={_QUESTION_WORD}\b)",
        text,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in raw_parts if p.strip()]

    if not parts:
        parts = [text]

    # if a part has no "?" and isn't a short follow-up, still keep it as its
    # own item rather than silently dropping it -- but a fragment left over
    # from meta-instruction stripping that's pure punctuation ("." with
    # nothing else) isn't a real sub-question and would just get sent to
    # generate_answer() as noise, so drop only that specific case.
    final: list[str] = []
    for p in parts:
        if p and not re.fullmatch(r"[.!?,\s]*", p):
            final.append(p)
    return final


def is_followup_only(question: str) -> bool:
    return any(p.match(question) for p in _FOLLOWUP_PATTERNS)


# ---------------------------------------------------------------------------
# Refusal / hedging detection
# ---------------------------------------------------------------------------

_REFUSAL_TEMPLATES = [
    "Informasi tersebut tidak ditemukan dalam dokumen yang tersedia.",
    "Informasi mengenai hal tersebut tidak ditemukan dalam dokumen yang diberikan.",
]
_refusal_template_embeddings = None  # lazy -- avoid loading the embedder at import time
_SEMANTIC_REFUSAL_THRESHOLD = 0.55


def _looks_like_refusal_semantic(answer: str) -> bool:
    global _refusal_template_embeddings
    import numpy as np

    embedder = retrieval.get_embedder()
    if _refusal_template_embeddings is None:
        _refusal_template_embeddings = embedder.encode(_REFUSAL_TEMPLATES, normalize_embeddings=True)
    answer_emb = embedder.encode([answer[:200]], normalize_embeddings=True)[0]
    return float(np.max(_refusal_template_embeddings @ answer_emb)) >= _SEMANTIC_REFUSAL_THRESHOLD


def looks_like_refusal(answer: str) -> bool:
    lower = answer.lower()
    if not any(phrase in lower for phrase in REFUSAL_PHRASES):
        return False
    # A refusal PHRASE appearing doesn't mean the answer IS a refusal --
    # confirmed live real domain content can legitimately say "not found"
    # in an unrelated sense (a real clinical "Tidak ditemukan kelainan
    # medis" category, correctly answered in detail). Keyword narrowing
    # and a length cutoff each traded one failure for worse ones (see
    # REFUSAL_PHRASES comment); semantic similarity to the system's own
    # actual refusal framing cleanly separated every real case tested
    # (genuine refusals 0.68-0.73, real content 0.16-0.44). Only runs when
    # a phrase already matched, so the common (no refusal language at all)
    # path never pays the embedder cost.
    return _looks_like_refusal_semantic(answer)


def looks_like_hedging(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in HEDGING_PHRASES)


# ---------------------------------------------------------------------------
# Single-question orchestration
# ---------------------------------------------------------------------------

def _passes_relevance_gate(hits: list[dict]) -> bool:
    # OR, not AND: the two signals fail on different questions (see
    # config.RELEVANCE_MIN_RERANK_SCORE), so clearing either one is real
    # signal that a hit is relevant. A missing rerank_score (RERANK_ENABLED
    # False) must fail closed, not fall back to the raw score -- that raw
    # score is on a 0-1 scale, RELEVANCE_MIN_RERANK_SCORE is calibrated for
    # the cross-encoder's unbounded logit scale, so a silent fallback would
    # make the OR trivially true for almost any hit.
    if not hits:
        return False
    max_raw = max(h["score"] for h in hits)
    rerank_scores = [h["rerank_score"] for h in hits if "rerank_score" in h]
    max_rerank = max(rerank_scores) if rerank_scores else float("-inf")
    return max_raw >= config.RELEVANCE_MIN_SCORE or max_rerank >= config.RELEVANCE_MIN_RERANK_SCORE


def generate_answer(
    question: str, session_id: Optional[str] = None, _forced_source: Optional[str] = None
) -> Union[Answer, ClarificationNeeded]:
    named_sources = [_forced_source] if _forced_source else retrieval.named_target_sources(question)

    # Multi-source discovery: this is a "find the file" system over a
    # corpus of unrelated real documents, not one document at a time -- a
    # question with no document named must never silently blend content
    # from several unrelated files into one paragraph (confirmed live: "ada
    # berapa bab?" with nothing named merged headings from a risk-management
    # doc and an unrelated SOP into one nonsense list). Look at how many
    # DISTINCT documents actually compete for the top retrieval slots and
    # branch: none -> not found; one -> just answer it, scoped to that one
    # automatically; a few -> answer each in full, clearly labeled by file,
    # never merged; too many to write out in full -> a short labeled
    # candidate list instead (cheap, no LLM call) and let the next message
    # naming one of them (typed, or the existing candidate-button flow)
    # come back through here with _forced_source set.
    if not named_sources:
        wide_hits = retrieval.retrieve(question, source_filter=None, k=config.MULTI_SOURCE_CANDIDATE_K)

        # Bare generic-identifier questions ("Apa isi Pasal 8?") with no
        # document named take a HARD clarification block here, never the
        # soft guess-plus-alternates path below -- a clause/lampiran number
        # is genuinely unanswerable without knowing whose numbering it is,
        # unlike a general topic question where guessing the best-ranked
        # document and offering alternates is still useful. Real bug fixed
        # here (caught by code review): detect_ambiguity() was previously
        # only ever called later in this function, in the branch that
        # requires named_sources to be non-empty -- but detect_ambiguity()
        # itself immediately returns None whenever named_sources IS set, so
        # it was unreachable dead code and every bare-identifier question
        # silently fell through to a guessed answer instead of asking which
        # document was meant.
        ambiguity = retrieval.detect_ambiguity(question, named_sources)
        if ambiguity is not None:
            message = (
                "Pertanyaan Anda merujuk pada identifikasi yang ditemukan di beberapa dokumen: "
                + ", ".join(ambiguity.candidates)
                + ". Mohon sebutkan dokumen yang dimaksud."
            )
            log_id = database.log_answer(session_id, question, message, None, False, False, True, wide_hits)
            return ClarificationNeeded(message=message, candidates=ambiguity.candidates, log_id=log_id)

        # Relevance gate (see _passes_relevance_gate): this is the path a
        # question with no named document takes -- confirmed live this is
        # where most off-topic questions were slipping through: vector
        # search always returns *some* top-k, several sources can end up
        # "competing" on the reranked score even when none are genuinely
        # relevant, and the ambiguity logic below only ever compares
        # candidates to each other, never asks whether any of them clear an
        # absolute floor at all.
        if not _passes_relevance_gate(wide_hits):
            # A query-rewrite fallback was tried here (rephrase into formal
            # register, retry retrieval) to rescue colloquial/short genuine
            # questions ("cara mengajukan cuti" vs. the document's formal
            # "Istirahat Tahunan") that fail both relevance signals.
            # Reverted: confirmed live in the real 100-question eval that
            # rewriting can inflate a genuinely off-topic question's scores
            # too (formal-register text matches generic corpus boilerplate
            # regardless of topic), and two different gating strategies for
            # the retry (OR-based, then rerank-only) each fixed one false
            # positive while breaking the original genuine rescue or another
            # off-topic case -- no threshold on these two signals reliably
            # separated a rewritten off-topic question from a rewritten
            # genuine one. This vocabulary-mismatch class of question
            # remains a known, deliberately unfixed gap: safer than
            # reopening any off-topic false-positive risk.
            answer = "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
            log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
            return Answer(text=answer, sources=[], log_id=log_id)

        # Group by source using the RERANKED score, not the raw cosine
        # score -- confirmed live this matters: raw cosine similarity alone
        # put two unrelated documents' boilerplate-matching chunks at
        # 0.28-0.41 (title-block and revision-table
        # boilerplate is nearly identical across unrelated documents),
        # enough to count as "competing" sources and wrongly trigger the
        # "too many documents" fallback below for a question that actually
        # had exactly one real answer. The cross-encoder rerank score
        # correctly told them apart (the one truly relevant chunk scored
        # +2.74, everything else -1.7 to -6.2) -- using it here is what
        # rerank() exists for in the first place (see its own docstring).
        def _relevance(h: dict) -> float:
            return h.get("rerank_score", h["score"])

        best_per_source: dict[str, dict] = {}
        for h in wide_hits:
            src = h["source"]
            if src not in best_per_source or _relevance(h) > _relevance(best_per_source[src]):
                best_per_source[src] = h
        ranked_sources = sorted(best_per_source.items(), key=lambda kv: _relevance(kv[1]), reverse=True)

        # A source only really "competes" if it comes within a reasonable
        # gap of the single best (reranked) score -- otherwise it's exactly
        # the kind of weak/coincidental match the reranker exists to filter
        # out, not genuine ambiguity about which document answers this.
        if ranked_sources:
            top_score = _relevance(ranked_sources[0][1])
            ranked_sources = [
                (src, hit) for src, hit in ranked_sources
                if top_score - _relevance(hit) <= config.MULTI_SOURCE_RERANK_GAP
            ]

        if not ranked_sources:
            answer = "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
            log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
            return Answer(text=answer, sources=[], log_id=log_id)

        if len(ranked_sources) == 1:
            return generate_answer(question, session_id=session_id, _forced_source=ranked_sources[0][0])

        # Multiple documents compete and none was named: answer once from
        # the highest-ranked one (a forced source always resolves to a real
        # Answer -- the bare-identifier ambiguity check above already
        # caught the "genuinely can't guess" case -- and one LLM call beats
        # generating a full answer per candidate), then note how many other
        # documents also looked relevant and offer them as candidates. The
        # UI shows the real answer immediately and the alternates as an
        # optional, dismissible "narrow it down" choice, never a hard block.
        top_src = ranked_sources[0][0]
        top_answer = generate_answer(question, session_id=session_id, _forced_source=top_src)
        alternates = ranked_sources[1 : config.MULTI_SOURCE_CANDIDATE_LIST_MAX + 1]
        note = (
            f"\n\nCatatan: pertanyaan ini juga cocok dengan {len(ranked_sources) - 1} dokumen lain yang "
            f"mungkin relevan. Jawaban di atas diambil dari dokumen dengan kecocokan tertinggi "
            f"({top_src}); pilih salah satu di bawah untuk fokus ke dokumen lain."
        )
        all_candidates = [top_src] + [src for src, _ in alternates]
        combined_text = top_answer.text + note
        # top_answer.log_id already points at an answer_log row written by
        # the recursive call above, but with just the inner single-source
        # text -- without this update the persisted row (and anything
        # reading it: flagging, an eval re-run against logged answers)
        # would silently disagree with what the user actually saw (caught
        # in code review, not live).
        if top_answer.log_id is not None:
            database.update_answer_text(top_answer.log_id, combined_text, session_id)
        return Answer(
            text=combined_text,
            sources=top_answer.sources,
            log_id=top_answer.log_id,
            candidates=all_candidates,
        )

    # Enumerate-intent routing: a "list all X" / "apa saja X" / "berapa
    # banyak X" question must never go through similarity retrieval, which
    # returns the most-relevant top-k, not a completeness guarantee (see
    # module docstring -- "list all Pasal" through the normal path returned
    # 4 of 22 real headings). Both deterministic paths read the real
    # heading/document index directly and format with zero LLM calls, so
    # the count can't be dropped by retrieval depth or invented by the model.
    broad = retrieval.broad_category_listing(question, named_sources)
    if broad is not None:
        lines = "\n".join(f"- {h}" for h in broad["headings"])
        answer = (
            f"Ditemukan {len(broad['headings'])} bagian '{broad['category']}':\n{lines}"
        )
        log_id = database.log_answer(session_id, question, answer, None, False, False, False, [])
        return Answer(text=answer, sources=broad["sources"], log_id=log_id)

    doc_listing = retrieval.document_type_listing(question, named_sources)
    if doc_listing is not None:
        lines = "\n".join(f"- {s}" for s in doc_listing["sources"])
        answer = (
            f"Ditemukan {len(doc_listing['sources'])} dokumen bertipe "
            f"'{doc_listing['doc_type']}':\n{lines}"
        )
        log_id = database.log_answer(session_id, question, answer, None, False, False, False, [])
        return Answer(text=answer, sources=doc_listing["sources"], log_id=log_id)

    # Full-document fallback: a broad question (no specific clause/section
    # identifier referenced) about exactly one small named document skips
    # retrieval and gets the whole document injected instead -- guarantees
    # completeness regardless of chunk-level retrieval quality. Two
    # conditions must both hold: it must fit this model's actual running
    # context window (real margin for the prompt template and response),
    # AND it must be small enough that full injection is actually reliable
    # (see config.FULL_DOC_FALLBACK_MAX_CHARS -- confirmed live that
    # "fits in context" isn't the same as "small enough to reason over
    # accurately"; a document can technically fit while still being too
    # large for the model to reliably find one specific fact inside it).
    # Otherwise falls through to the normal retrieval path, which finds
    # the single most relevant passage directly instead of confusing it
    # with everything else in a large document.
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
            # Same fabrication check the normal retrieval path already
            # runs (see below) -- confirmed live this path was missing it
            # entirely: a full-document answer fabricated a plausible-
            # looking technical detail ("HTTP 401") that wasn't actually
            # in the document. Applies to any claimed identifier/code
            # against this document's own real headings, not specific to
            # any one document or fact type.
            headings = retrieval.get_headings_for(named_sources)
            unverified = unverified_identifiers(answer, headings)
            footer_notes = ["Dihasilkan dari keseluruhan isi dokumen, bukan potongan teks."]
            if unverified:
                footer_notes.append(
                    f"Catatan: {', '.join(unverified)} belum terverifikasi terhadap dokumen sumber."
                )
            answer += _build_footer(footer_notes)
            log_id = database.log_answer(
                session_id, question, answer, None, looks_like_hedging(answer),
                looks_like_refusal(answer), False, [],
                prompt_tokens=prompt_tokens, response_tokens=response_tokens,
            )
            return Answer(text=answer, sources=[named_sources[0]], log_id=log_id)

    # No RELEVANCE_MIN_SCORE gate here (unlike the no-named-document path
    # above): named_sources is always non-empty by this point (the empty
    # case already returned above), so the user explicitly named this
    # document -- there's no "nothing in the whole corpus is relevant"
    # risk to guard against. Confirmed live: all 13 off-topic false-
    # positives in the original 44%-refusal bug had NO named document; a
    # gate added here anyway wrongly refused 12/60 (20%) of real named-
    # document answerable questions at this path's real k=8, because a
    # single filtered document's per-chunk cosine scores span a much wider
    # range (0.42-1.0) than the whole-corpus noise floor this threshold
    # was calibrated against.
    # detect_ambiguity() itself only ever fires when named_sources is EMPTY
    # (see its own docstring/guard in retrieval.py) -- calling it here, where
    # named_sources is always non-empty by this point, always returned None.
    # Moved to the actual no-named-document branch above, where it can fire.
    source_filter = named_sources or None
    hits = retrieval.retrieve(question, source_filter=source_filter, k=config.TOP_K)

    if not hits:
        answer = "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang tersedia."
        log_id = database.log_answer(session_id, question, answer, 0.0, False, True, False, [])
        return Answer(text=answer, sources=named_sources, log_id=log_id)

    prompt = build_prompt(question, hits)
    answer, prompt_tokens, response_tokens = call_ollama(prompt)

    relevant_sources = list({h["source"] for h in hits})
    headings = retrieval.get_headings_for(relevant_sources)
    unverified = unverified_identifiers(answer, headings, {h["page"] for h in hits})
    footer_notes = []
    if unverified:
        footer_notes.append(
            f"Catatan: {', '.join(unverified)} belum terverifikasi terhadap dokumen sumber."
        )

    avg_score = sum(h["score"] for h in hits) / len(hits)

    ocr_confidences = [h["ocr_confidence"] for h in hits if h.get("ocr_confidence") is not None]
    avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
    if avg_ocr_confidence is not None and avg_ocr_confidence < database.LOW_OCR_CONFIDENCE_THRESHOLD:
        footer_notes.append(
            f"Catatan: sebagian konteks berasal dari hasil pemindaian (OCR) dengan keyakinan rendah "
            f"(~{avg_ocr_confidence:.0f}/100)."
        )

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


_RESUME_WITH_DOCUMENT_RE = re.compile(r"^(.*?)\s+untuk dokumen\s+(\S.*)$", re.IGNORECASE | re.DOTALL)


def generate_multi_answer(text: str, session_id: Optional[str] = None) -> list[dict]:
    """Runs each sub-question through generate_answer(). Every sub-question
    appears in the output, explicitly, even if it couldn't be answered --
    never silently dropped.

    A resumed clarification ("<original question> untuk dokumen <file>",
    composed by the UI's candidate buttons or typed by an API caller
    resolving a multi-source candidate list) must be treated as ONE
    question with a forced source, never run through split_questions() --
    the original question almost always ends in "?", which split_questions()
    reads as a sentence boundary and splits the document name off into its
    own meaningless fragment, so the forced-source connection is lost and
    the LLM ends up answering neither part correctly (confirmed live: this
    silently degraded into a generic "what is this document about" answer
    instead of resuming the actual original question)."""
    stripped = text.strip()
    resume_match = _RESUME_WITH_DOCUMENT_RE.match(stripped)
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
                    "answer": f"[Terjadi kesalahan saat memproses pertanyaan ini: {exc}]",
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
            # looks_like_refusal() now pays a real embedder call on a
            # keyword-phrase match (see its own docstring) -- computing it
            # once instead of twice here avoids doubling that cost on every
            # refusal-shaped answer.
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
