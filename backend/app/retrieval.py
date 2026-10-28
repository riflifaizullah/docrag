"""Embedding-based retrieval, scoped heading lookups, named-document
targeting, ambiguity detection, and deterministic broad-category listing.

Carries fixes for known-bugs #1, #2, #7 from the rebuild spec:
  #1 retrieve() takes an optional source_filter and turns it into a Chroma
     `where` clause -- naming a document actually constrains the k-NN search.
  #2 heading lookups are always scoped to specific source(s), never the
     whole corpus.
  #7 broad-category word selection tries candidates in the order they
     appear in the question, not via unordered set iteration.

Later additions, layered onto the same retrieve() pipeline in this order:
  - reranking (get_reranker()/rerank()): fetch a wider embedding candidate
    set, then have a cross-encoder score (question, chunk) pairs jointly --
    catches cases where several near-identical boilerplate clauses tie on
    pure embedding similarity but only one is topically on point.
  - exact-identifier boost (_exact_identifier_boost()): overrides both of
    the above outright when the question names a specific clause/section
    number, because dense retrieval still doesn't key strongly on a literal
    number difference between otherwise-similar clauses.
  - own-document-code boost (_own_document_code_boost()): when a question
    is scoped to exactly one document, ensures a chunk containing that
    document's OWN full document number (derived from its filename, the
    same real numbering convention every indexed file follows) is present
    in the retrieved set. Confirmed live: a chunk from a DIFFERENT document
    referenced inside the scoped one (e.g. a related policy listed in a
    "Daftar Lampiran" appendix, with an incomplete/placeholder-style
    number) out-ranked the scoped document's own real title-block chunk --
    both are short, similarly-formatted "NOMOR / BERLAKU TMT" snippets the
    cross-encoder can't tell apart from text alone, since doing so needs
    document-structure awareness (this is the cover page vs. this is an
    appendix entry), not just topical similarity.
  - the shorthand-matching regex (_build_shorthand_patterns()), the
    identifier-boost pattern, and the broad-category word detection
    (_category_candidates_from_headings()) are all built from real indexed
    content (filenames / retrieved chunk text / real headings) rather than
    a hardcoded vocabulary, on the theory that anything corpus-specific
    (document-type words, section-keyword words) will silently go stale
    the moment this is ported to a differently-named real document set.
    GENERIC_IDENTIFIER_PATTERN (ambiguity detection) has NOT been
    generalized the same way yet -- still a fixed (pasal|lampiran|bab)
    list, same latent risk, just not hit by a real test case so far.
  - enumerate-intent routing (broad_category_listing(), document_type_
    listing()): "list all X" / "apa saja X" / "berapa banyak X" questions
    are answered by reading the real heading/document index directly and
    formatting deterministically, never through retrieve() at all --
    similarity search only returns the *most relevant* top-k chunks, never
    a completeness guarantee, so an enumerate question routed through it
    can (and did, live) silently return a partial list.
  - hybrid search (_bm25_search(), reciprocal rank fusion in retrieve()):
    a sparse/keyword BM25 pass runs alongside the embedding pass for the
    "specific"/"broad" paths (enumerate bypasses retrieval entirely), since
    embeddings alone can rank a chunk with the wrong literal clause number
    as more similar than the chunk with the right one on short,
    structurally-repetitive legal text.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app import config, database

_COLLECTION_NAME = "documents"

_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None
_reranker: Optional[CrossEncoder] = None
_reranker_unavailable = False

_sources_cache = database.TTLCache(config.FILENAME_CACHE_TTL_SECONDS)
_bm25_index_cache: dict = {}

# ChromaDB's embedded PersistentClient spins up internal background
# threads (its pubsub/telemetry subsystem) tied to whatever OS thread
# creates and first uses it. FastAPI's sync `def` endpoints run on a
# rotating threadpool (a different worker thread per request, not
# guaranteed to be the same one) -- confirmed live: a question that always
# succeeded via a plain single-threaded script failed 100% of the time
# through the API with a bare `KeyError` ("Label not found" spam from the
# client's internal consumer threads misrouting). Funneling every
# Chroma-touching call through one dedicated, always-the-same thread
# fixes this at the root instead of retrying around it.
_chroma_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-io")


def run_on_chroma_thread(fn, *args, **kwargs):
    return _chroma_thread.submit(fn, *args, **kwargs).result()


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=_DEVICE)
    return _embedder


def get_reranker() -> Optional[CrossEncoder]:
    """Lazily loads the cross-encoder. Explicit device=_DEVICE because,
    unlike SentenceTransformer, CrossEncoder was confirmed live to NOT
    reliably auto-select CUDA on its own -- it silently loaded onto CPU here
    even with a working CUDA-enabled torch and an available GPU, which
    matters since reranking runs on every retrieval call. If loading still
    fails (OOM, download failure), retrieval must still work -- degrade to
    embedding-only ranking rather than fail the request."""
    global _reranker, _reranker_unavailable
    if _reranker_unavailable:
        return None
    if _reranker is None:
        try:
            _reranker = CrossEncoder(config.RERANK_MODEL, device=_DEVICE)
        except Exception:
            _reranker_unavailable = True
            return None
    return _reranker


def rerank(question: str, hits: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder reranking of the wider embedding-candidate set fetched
    by retrieve(). An embedding comparing the question and a chunk
    independently often can't tell two near-identical boilerplate clauses
    apart; a cross-encoder scores the (question, chunk) pair jointly and
    is measurably better at picking the one that's actually on-topic (see
    the mmarco-mMiniLMv2 A/B check: an off-topic clause that tied for 2nd
    place under pure embedding similarity got correctly demoted here)."""
    if not hits:
        return hits
    reranker = get_reranker()
    if reranker is None:
        return hits[:top_k]
    pairs = [(question, h["text"]) for h in hits]
    scores = reranker.predict(pairs)
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def list_indexed_sources() -> list[str]:
    """Short-TTL cached listing -- avoids recomputing per-call within one
    request (known-bug: called 3x per question added seconds of overhead)."""
    return _sources_cache.get_or_set(database.list_document_sources)


def invalidate_sources_cache() -> None:
    _sources_cache.invalidate()
    _bm25_index_cache.clear()  # BM25 indexes are stale the moment the corpus changes


# ---------------------------------------------------------------------------
# Named / shorthand document targeting
# ---------------------------------------------------------------------------

_CONNECTOR = r"(?:(?:no\.?|nomor|number)\s+)?"

# Contract-type words ("Konsultasi", "Jasa", ...) sitting between "kontrak"
# and the number ("kontrak konsultasi 104") broke the shorthand match the
# same way "kontrak nomor 106" did before it -- same root cause, a rigid
# `noun <number>` pattern with no tolerance for a word in between. Rather
# than hardcode a fixed list of type words (which would not necessarily
# match every corpus's naming convention), the vocabulary is
# parsed from the filenames actually indexed right now, the same principle
# `heading_lead_words()` uses for identifier grounding: derive the
# recognized words from real content instead of a fixed list.
_CONTRACT_FILENAME_TYPE_RE = re.compile(r"^kontrak[_\s]+([a-zA-Z]+)[_\s]+\d", re.IGNORECASE)


def _contract_type_words(sources: list[str]) -> set[str]:
    words = set()
    for source in sources:
        m = _CONTRACT_FILENAME_TYPE_RE.match(source)
        if m:
            words.add(m.group(1).lower())
    return words


def _build_shorthand_patterns(sources: list[str]) -> list[re.Pattern]:
    type_words = _contract_type_words(sources)
    type_group = ""
    if type_words:
        type_group = r"(?:(?:" + "|".join(re.escape(w) for w in sorted(type_words)) + r")\s+)?"
    # tolerates, in any combination: a bare number ("kontrak 104"), a
    # connector word ("kontrak nomor 104"), and/or a real contract-type
    # word pulled from the indexed corpus ("kontrak konsultasi 104",
    # "kontrak konsultasi nomor 104") -- instead of requiring the number to
    # immediately follow the noun.
    return [
        re.compile(r"\b(contract|kontrak)\s+" + type_group + _CONNECTOR + r"(\d+)", re.IGNORECASE),
        re.compile(r"\b(sop)\s+" + _CONNECTOR + r"(\d+)", re.IGNORECASE),
    ]


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_fuzzy_match(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def named_target_sources(question: str) -> list[str]:
    """Returns real indexed filenames referenced in the question, in the
    order the reference appears in the question text (first-named noun
    wins -- see known-bug #7 for why unordered matching is unsafe)."""
    sources = list_indexed_sources()
    matches: list[tuple[int, str]] = []  # (position, source)

    # 1. exact filename mention (with or without .pdf)
    for source in sources:
        stem = source[:-4] if source.lower().endswith(".pdf") else source
        for needle in (source, stem):
            pos = question.lower().find(needle.lower())
            if pos != -1:
                matches.append((pos, source))
                break

    # 2. shorthand mentions: "contract 250", "kontrak 250", "sop 45",
    # "kontrak konsultasi 104", "kontrak nomor 106", etc.
    for pattern in _build_shorthand_patterns(sources):
        for m in pattern.finditer(question):
            number = m.group(2)
            candidates = [s for s in sources if number in s]
            for c in candidates:
                matches.append((m.start(), c))

    # 3. fuzzy filename mention: real filenames are long, punctuation-heavy
    # codes ("1._DOC-017_2024_Rev_00_...pdf") that no one types
    # verbatim from memory -- typing the same words with different
    # underscores/spaces/brackets shouldn't fall through to a fuzzy,
    # corpus-wide search when the user was clearly naming one specific
    # document (confirmed live: a near-exact retyping of a real filename,
    # differing only in separators, missed the exact-match check above and
    # fell through to multi-source discovery, which then pulled in other
    # documents that merely CITE this one in a references list -- a
    # precision problem caused by never recognizing the document was
    # actually named). Comparing on a punctuation-stripped normal form
    # catches this without any per-corpus vocabulary. Only applied to
    # filenames with enough real content (not just a bare number) to make
    # a coincidental match implausible.
    if not matches:
        norm_question = _normalize_for_fuzzy_match(question)
        for source in sources:
            stem = source[:-4] if source.lower().endswith(".pdf") else source
            norm_stem = _normalize_for_fuzzy_match(stem)
            if len(norm_stem) >= 20 and norm_stem in norm_question:
                matches.append((norm_question.find(norm_stem), source))

    if not matches:
        return []

    matches.sort(key=lambda t: t[0])
    seen = set()
    ordered: list[str] = []
    for _, src in matches:
        if src not in seen:
            seen.add(src)
            ordered.append(src)
    return ordered


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_IDENTIFIER_EXTRACT_RE = re.compile(
    # any capitalized word (or hyphen/space-joined run of them) followed by
    # an alphanumeric identifier -- no fixed keyword list, so this covers
    # "Ayat 11.1", "Pasal 11", "Lampiran E1", "Tanggung Jawab - Sub 4",
    # "Referensi - Sub 9", or whatever the next document type phrases its
    # identifiers as, the same way heading_lead_words() in generation.py
    # builds its vocabulary from real content instead of a hardcoded list.
    # [ \t] (not \s): a real identifier is always written on one line, so
    # the separator must not also match a newline -- see generation.py's
    # _IDENTIFIER_PATTERN for the confirmed-live bug this mirrors.
    r"\b([A-Z][a-zA-Z]*(?:[ \t-]+[A-Z][a-zA-Z]*)*)[ \t]+([A-Za-z]?\d+(?:[.\-]\d+)?)\b"
)


def _exact_identifier_boost(question: str, source_filter: Optional[list[str]]) -> Optional[dict]:
    """Pure embedding search doesn't key strongly on a literal clause number
    (e.g. "Ayat 11.1", or a SOP's "Tanggung Jawab - Sub 4") when the document
    is full of near-identical boilerplate clauses -- confirmed on the
    regenerated, denser corpus, where the target clause sometimes fell
    outside top-k. When the question names a specific document AND a
    specific identifier, score each of that document's chunks by how much
    of the identifier phrase it actually contains and let the best one win
    outright -- an exact "phrase identifier" substring (contract-style)
    wins immediately; a compound reference whose wording doesn't appear
    verbatim in the source text (the SOP's "Sub-bagian N" body phrasing vs.
    a question asking about "Sub N") still wins by matching most of the
    phrase's distinguishing words plus the identifier itself, so the fix
    doesn't depend on the document's exact surface wording."""
    if not source_filter:
        return None
    match = _IDENTIFIER_EXTRACT_RE.search(question)
    if not match:
        return None
    phrase, ident = match.group(1), match.group(2)
    phrase_words = [w for w in re.split(r"[\s-]+", phrase) if w]
    if not phrase_words:
        return None
    ident_re = re.compile(r"\b" + re.escape(ident.lower()) + r"\b")
    exact_needle = f"{phrase} {ident}".lower()
    min_matched = max(1, len(phrase_words) - 1)  # tolerate one non-matching word (e.g. a connector)

    collection = get_collection()
    where = {"source": {"$in": source_filter}} if len(source_filter) > 1 else {"source": source_filter[0]}
    all_chunks = collection.get(where=where, include=["documents", "metadatas"])

    best_score = None
    best = None
    for doc, meta in zip(all_chunks.get("documents", []), all_chunks.get("metadatas", [])):
        doc_lower = doc.lower()
        if not ident_re.search(doc_lower):
            continue
        matched = sum(1 for w in phrase_words if w.lower() in doc_lower)
        if matched < min_matched:
            continue
        exact = exact_needle in doc_lower
        score = (exact, matched, -len(doc))
        if best_score is None or score > best_score:
            best_score = score
            best = (doc, meta)

    if best is None:
        return None
    doc, meta = best
    return {
        "text": doc,
        "source": meta.get("source"),
        "page": meta.get("page"),
        "distance": 0.0,
        "score": 1.0,
        "ocr_confidence": _meta_ocr_confidence(meta),
    }


_QUOTED_SPAN_RE = re.compile(r"['‘’\"“”]([^'‘’\"“”]{3,})['‘’\"“”]")


def _find_quoted_heading_pages(question: str, headings: list[dict]) -> list[int]:
    """Match a real heading against a question asking for the content of
    that heading by name (e.g. "Apa isi bagian '<heading>' pada dokumen
    ..."), and return ALL pages that share the single best-matching heading
    text. Confirmed live: pure similarity retrieval keeps ranking the
    HEADING TEXT ITSELF above the real body content that follows it,
    because the question quotes the heading almost verbatim while the body
    text uses different wording -- on a large, heading-dense document, that
    body content can fall out of top-k entirely even after deduplicating
    the heading's own repeated occurrences (see _dedupe_repeated_text). A
    running header repeated across a multi-page attachment (confirmed live:
    the same heading text on 9 different pages of one real document) means
    the real content could be on ANY of those pages, not just the first --
    collecting all of them, not one arbitrary match, is what actually
    surfaces it. Once the question is confirmed to be asking about a
    specific, already-known real heading, bypass similarity ranking for it
    entirely -- same principle as broad_category_listing()/
    document_type_listing() reading the index directly for other "look it
    up, don't rank for it" question shapes.

    Only matches against an actual QUOTED span in the question, not the
    whole question text -- confirmed live (independent review + real
    corpus check) that matching against the whole question let a bare,
    generic 2-word heading ("RUANG LINGKUP", "TATA CARA" -- 292 distinct
    examples exist in this corpus, including "DAFTAR ISI") hijack retrieval
    to an arbitrary page on ANY question mentioning those common words,
    with no quoting intent at all. The real question shape always quotes
    the heading; requiring that is what actually distinguishes "asking
    about this specific heading" from a question that merely uses common
    words. Still requires most of the heading's own distinguishing words
    to appear within the quoted span, not just one -- same false-positive
    guard _exact_identifier_boost already uses (min_matched tolerance) so
    a short/generic heading doesn't match a loosely-related quote."""
    quoted_spans = [m.group(1).lower() for m in _QUOTED_SPAN_RE.finditer(question)]
    if not quoted_spans:
        return []
    best_text = None
    best_ratio = 0.0
    for h in headings:
        words = [w for w in re.split(r"\s+", h["text"].strip()) if len(w) > 2]
        if len(words) < 2:
            continue
        for span in quoted_spans:
            matched = sum(1 for w in words if w.lower() in span)
            ratio = matched / len(words)
            if ratio > best_ratio:
                best_ratio = ratio
                best_text = h["text"]
    if best_text is None or best_ratio < 0.7:
        return []
    return sorted({h["page"] for h in headings if h["text"] == best_text})


def _heading_quote_boost(question: str, source_filter: Optional[list[str]], k: int) -> list[dict]:
    if not source_filter or len(source_filter) != 1:
        return []
    pages = _find_quoted_heading_pages(question, get_headings_for(source_filter))
    if not pages:
        return []
    collection = get_collection()
    where = {"$and": [{"source": source_filter[0]}, {"page": {"$in": pages}}]}
    got = collection.get(where=where, include=["documents", "metadatas"])
    out = []
    for doc, meta in zip(got.get("documents", []), got.get("metadatas", [])):
        out.append({
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": 0.0,
            "score": 1.0,
            "ocr_confidence": _meta_ocr_confidence(meta),
        })
    return out[:k]


def _meta_ocr_confidence(meta: dict) -> Optional[float]:
    val = meta.get("ocr_confidence")
    return None if val is None or val < 0 else val


# Every real indexed file follows the same document-numbering convention
# regardless of type (SOP, policy, contract): a letter-dash-digits
# code, a department code, a 4-digit year, and an "S" revision suffix
# (e.g. "A-035-DEPT3000-2025-S9", "B-020-DEPT0400-2023-S9",
# "C-026-DEPT1310-2026-S9") -- derived structurally from that observed
# format across many real different filenames, not one hardcoded code.
_DOC_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z])[\s\-–]*(\d+)[\s\-–/]+([A-Za-z0-9]+)[\s\-–/]+"
    r"(\d{4})[\s\-–]*S[\s]*(\d+)(?![A-Za-z0-9])"
)
# Confirmed live this needed to be looser than the filename convention
# alone: real document TEXT can render the same code with different
# separators than its filename does -- e.g. a filename's "A-035-..." (a
# plain hyphen, no spaces) appeared inside one real document's own content
# as "C – 035/..." (an en-dash WITH spaces around it). Both the letter-
# digit separator and the segment separators tolerate whitespace/hyphen/
# en-dash so either real rendering matches, while still requiring the
# full structural shape (letter, digits, alnum code, 4-digit year, "S" +
# digits) -- a placeholder/incomplete reference ("A-..../DEPT3000/2025-S9",
# confirmed live as the wrong-document case _own_document_code_boost()
# exists to exclude) still correctly fails to match, since dots aren't
# digits.


def _source_document_code(source: str) -> Optional[str]:
    match = _DOC_CODE_RE.search(source)
    if not match:
        return None
    # Normalize the whole matched span (not the groups reassembled without
    # their original separators) so a filename's dashes and an in-document
    # mention's slashes/dashes normalize to the exact same token structure,
    # not just the same characters in a different arrangement.
    return _normalize_for_fuzzy_match(match.group(0))


def _own_document_code_boost(source_filter: Optional[list[str]]) -> list[dict]:
    """When a question is scoped to exactly one document, make sure a chunk
    containing that document's OWN full document number is present in the
    retrieved set -- confirmed live this matters: a chunk from a DIFFERENT
    document referenced inside the scoped one (a related policy listed in
    an appendix, with an incomplete/placeholder-style number like
    "A-..../DEPT3000/2025-S9") out-scored the scoped document's own real
    title-block chunk under reranking, since both are short, similarly-
    formatted metadata snippets. The scoped document's real number is
    derived from its own filename (the corpus's real, observed numbering
    convention), not guessed -- a chunk containing that number IN FULL is
    strong direct evidence it's genuinely this document's own metadata,
    not merely a mention of a similarly-formatted different one (whose
    number, being different, won't match).

    A large document can have several scattered chunks that each contain
    the own-code (confirmed live: a stray page-17 remnant of the running
    header carried just the "NOMOR : ..." line with no other field, having
    lost the rest to _strip_repeating_page_lines() elsewhere) -- picking
    the wrong one wins the number-match but still misses whatever field the
    question actually needs. A document's real cover/title-block metadata
    is essentially always on its first few pages, never deep in the body,
    so among every chunk containing the own-code, the one on the lowest
    page number is preferred -- a general structural fact about how
    business documents are laid out, not specific to any one document.

    A title-block metadata table can also be split across a chunk boundary
    (confirmed live: a real table's own document-number field ended one
    chunk, with its BERLAKU TMT/JUDUL/HALAMAN fields continuing into the
    next chunk, which doesn't repeat the number and so wouldn't match on
    its own) -- the chunk immediately following the selected one (by
    page/chunk_index order, same source) is included too, so a split
    table's other half stays available. This is a generic char-count
    chunking-boundary phenomenon that can happen to any short metadata
    table landing near the chunk-size cutoff, not specific to any one
    document's content."""
    if not source_filter or len(source_filter) != 1:
        return []
    own_code = _source_document_code(source_filter[0])
    if not own_code:
        return []

    collection = get_collection()
    all_chunks = collection.get(where={"source": source_filter[0]}, include=["documents", "metadatas"])
    entries = [
        (meta.get("page") or 0, meta.get("chunk_index") or 0, doc, meta)
        for doc, meta in zip(all_chunks.get("documents", []), all_chunks.get("metadatas", []))
    ]
    candidates = [e for e in entries if own_code in _normalize_for_fuzzy_match(e[2])]
    if not candidates:
        return []
    best = min(candidates, key=lambda e: (e[0], e[1]))

    def _to_hit(entry: tuple) -> dict:
        _, _, doc, meta = entry
        return {
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": 0.0,
            "score": 1.0,
            "ocr_confidence": _meta_ocr_confidence(meta),
        }

    results = [_to_hit(best)]
    later = sorted((e for e in entries if (e[0], e[1]) > (best[0], best[1])), key=lambda e: (e[0], e[1]))
    if later:
        results.append(_to_hit(later[0]))
    return results


_TITLE_BLOCK_FIELD_RE = re.compile(r"[^\s:]{2,20}\s*:\s*\S")


def find_foreign_document_codes(source: str) -> list[dict]:
    """Audits ONE document for a real, confirmed-live anomaly: a page whose
    own title-block structurally self-identifies as a DIFFERENT document
    than this file's own filename/cover implies -- confirmed live on one
    real corpus file, whose filename and cover page claim one document
    (C-035) but whose page 3 is an intact title-block table for a
    completely different document (C-023). This is a source-data mixing
    problem (wrong pages assembled into the PDF at the source), not
    something ingestion/retrieval code can fix -- this only detects and
    reports it for human review.

    A real citation of another document's number in ordinary body prose
    (a REFERENSI list) is common and NOT flagged -- only a foreign code
    found inside its own short, title-block-SHAPED chunk (multiple
    colon-separated "Label : Value" fields, the same structural signature
    used elsewhere in this file for detecting real title-blocks) counts,
    since a real citation reads as a sentence, not a standalone field
    table.

    Not every real filename embeds its own document code (confirmed live:
    the exact file this was built to catch, "27._Portal_Guide.pdf",
    is named descriptively, not by code, so _source_document_code() on the
    filename alone returns None and this check would silently never fire
    for it) -- when the filename doesn't give a code, the code found in the
    lowest-page title-block-shaped chunk is used as the reference instead,
    since a document's own real identity is reliably on its earliest pages
    (the same assumption _own_document_code_boost() already relies on).

    Known open limitation, confirmed live: on the exact motivating file
    above, this still doesn't fire, because that file's own cover page has
    NO colon-separated fields at all (just plain title lines), so
    _is_title_block() never recognizes ANY of its chunks as a title-block
    -- own_code stays None and the whole check returns empty. Detecting a
    plain-title cover (no colons) as this document's identity anchor would
    need a different, not-yet-designed signal; flagged as a real gap
    rather than silently claimed as solved."""
    collection = get_collection()
    all_chunks = collection.get(where={"source": source}, include=["documents", "metadatas"])
    entries = list(zip(all_chunks.get("documents", []), all_chunks.get("metadatas", [])))

    def _is_title_block(doc: str) -> bool:
        return len(doc) <= 400 and len(_TITLE_BLOCK_FIELD_RE.findall(doc)) >= 2

    own_code = _source_document_code(source)
    if own_code is None:
        title_block_entries = sorted(
            (e for e in entries if _is_title_block(e[0])),
            key=lambda e: (e[1].get("page") or 0, e[1].get("chunk_index") or 0),
        )
        for doc, _ in title_block_entries:
            m = _DOC_CODE_RE.search(doc)
            if m:
                own_code = _normalize_for_fuzzy_match(m.group(0))
                break
    if own_code is None:
        return []  # no way to establish this document's own identity at all

    findings = []
    for doc, meta in entries:
        if not _is_title_block(doc):
            continue
        for match in _DOC_CODE_RE.finditer(doc):
            code = _normalize_for_fuzzy_match(match.group(0))
            if code != own_code:
                findings.append(
                    {
                        "page": meta.get("page"),
                        "chunk_index": meta.get("chunk_index"),
                        "found_code": match.group(0),
                        "text": doc,
                    }
                )
    return findings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_cache_key(source_filter: Optional[list[str]]):
    return tuple(sorted(source_filter)) if source_filter else None


def _get_bm25_index(source_filter: Optional[list[str]]):
    """One BM25 index per distinct source_filter scope, invalidated whenever
    the corpus changes (see invalidate_sources_cache()) -- built once per
    scope on demand and cached, not rebuilt per query, so this cost is paid
    once per corpus change, not once per question.

    The fetch itself is paginated (config.CHROMA_PAGE_SIZE, the same
    pattern database.list_document_sources() already uses for SQLite) --
    at real scale (thousands of PDFs, tens of thousands of chunks) a single
    unpaginated collection.get() risks the same "too many SQL variables"
    style limit CHROMA_PAGE_SIZE already exists elsewhere in this codebase
    to avoid; this only ever showed up as "cheap" against the 20-file test
    corpus this system was originally built and verified against."""
    key = _bm25_cache_key(source_filter)
    if key in _bm25_index_cache:
        return _bm25_index_cache[key]

    collection = get_collection()
    where = None
    if source_filter:
        where = {"source": {"$in": source_filter}} if len(source_filter) > 1 else {"source": source_filter[0]}

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    offset = 0
    while True:
        fetched = collection.get(
            where=where,
            include=["documents", "metadatas"],
            limit=config.CHROMA_PAGE_SIZE,
            offset=offset,
        )
        page_ids = fetched.get("ids", [])
        if not page_ids:
            break
        ids.extend(page_ids)
        docs.extend(fetched.get("documents", []))
        metas.extend(fetched.get("metadatas", []))
        offset += config.CHROMA_PAGE_SIZE

    entry = None if not docs else (BM25Okapi([_tokenize(d) for d in docs]), ids, docs, metas)
    _bm25_index_cache[key] = entry
    return entry


def _bm25_search(question: str, source_filter: Optional[list[str]], k: int) -> list[dict]:
    """Sparse/keyword retrieval pass alongside the embedding pass. On short,
    structurally-repetitive legal text, an embedding can rank the chunk with
    the exact WRONG Pasal number as more similar than the chunk with the
    right one -- every clause uses near-identical wording, so the literal
    number is often the only real signal. BM25 catches that literal match
    directly; results get combined with the embedding ranking via
    reciprocal rank fusion in retrieve() rather than trusted alone."""
    entry = _get_bm25_index(source_filter)
    if entry is None:
        return []
    bm25, ids, docs, metas = entry
    scores = bm25.get_scores(_tokenize(question))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {"id": ids[i], "text": docs[i], "meta": metas[i]}
        for i in ranked_idx
        if scores[i] > 0
    ]


def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _dedupe_repeated_text(ordered_ids: list[str], hits_by_id: dict[str, dict]) -> list[str]:
    # A running page header repeated verbatim across a multi-page
    # attachment can occupy most of top-k on a query that quotes the
    # heading -- confirmed live this crowded out the one chunk (a
    # different page of the SAME document) that had the real body content,
    # producing an honest but wrong "not found" despite the answer being
    # in the corpus. Only same-source repeats collapse: two different
    # documents sharing boilerplate text is a different, unrelated case.
    seen: set[tuple] = set()
    out = []
    for hid in ordered_ids:
        h = hits_by_id[hid]
        key = (h["source"], h["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(hid)
    return out


def _apply_boost(hits: list[dict], boost_items: list[dict], k: int) -> list[dict]:
    """Prepend forced boost items ahead of the normally-ranked hits,
    respecting the k budget on both the boost itself and the combined
    result. Confirmed live (independent code review): the previous inline
    formula (`boost + hits[:max(0, k-len(boost))]`) never capped the boost
    list itself, so a single boost bigger than k -- or several boosts
    stacking, each prepending onto whatever the last one left -- could
    make the final result exceed k entirely, silently blowing the budget
    every other caller of retrieve() assumes is respected."""
    return (boost_items[:k] + hits)[:k]


def retrieve(question: str, source_filter: Optional[list[str]] = None, k: int = config.TOP_K) -> list[dict]:
    collection = get_collection()
    embedder = get_embedder()
    query_embedding = embedder.encode([question], show_progress_bar=False).tolist()

    where = None
    if source_filter:
        where = {"source": {"$in": source_filter}} if len(source_filter) > 1 else {"source": source_filter[0]}

    candidate_k = max(k, config.RERANK_CANDIDATE_K) if config.RERANK_ENABLED else k
    if where:
        # Confirmed live: chromadb's HNSW query raises RuntimeError
        # ("Cannot return the results in a contigious 2D array. Probably
        # ef or M is too small") when n_results exceeds the number of
        # vectors actually matching a `where` filter -- a filtered search
        # can't return more results than exist in the filtered subset. A
        # source-scoped query (a specific named document) routinely hits
        # this: many real documents in this corpus have far fewer than
        # RERANK_CANDIDATE_K (100) chunks. Not a corner case -- any question
        # naming a short document would crash in production, not just in
        # a synthetic eval. Cap n_results to the real filtered count first
        # (a cheap id-only get, no document/embedding payload) rather than
        # discovering the ceiling by crashing.
        try:
            filtered_ids = collection.get(where=where, include=[]).get("ids", [])
        except Exception:
            filtered_ids = None
        if filtered_ids is not None:
            candidate_k = max(1, min(candidate_k, len(filtered_ids)))
    # The count-based cap above still isn't airtight -- confirmed live in a
    # 100-question eval, one filtered query still hit the same HNSW
    # RuntimeError despite candidate_k <= the real filtered count. HNSW's
    # filtered search is an approximate graph traversal, not an exact
    # filter-then-scan: it isn't guaranteed to surface `candidate_k` results
    # even when that many technically exist in the index. Halving and
    # retrying is the honest fix for an approximate algorithm's edge case --
    # down to 1, which is always satisfiable, so this can't loop forever.
    while True:
        try:
            result = collection.query(
                query_embeddings=query_embedding,
                n_results=candidate_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            break
        except RuntimeError as exc:
            if "contigious 2D array" not in str(exc) or candidate_k <= 1:
                raise
            candidate_k = max(1, candidate_k // 2)

    embedding_ids = result.get("ids", [[]])[0]
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    hits_by_id: dict[str, dict] = {}
    embedding_rank: dict[str, int] = {}
    for rank, (hid, doc, meta, dist) in enumerate(zip(embedding_ids, docs, metas, dists)):
        embedding_rank[hid] = rank
        hits_by_id[hid] = {
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": dist,
            "score": 1 - dist,  # cosine distance -> similarity-ish score; preserved as-is
            # for anything comparing scores across hits (e.g. ambiguity
            # detection's score-gap check) -- RRF fusion below only decides
            # which candidates advance to reranking, it never overwrites
            # this field's meaning.
            "ocr_confidence": _meta_ocr_confidence(meta),
        }

    bm25_rank: dict[str, int] = {}
    if config.HYBRID_SEARCH_ENABLED:
        bm25_hits = _bm25_search(question, source_filter, config.BM25_CANDIDATE_K)
        bm25_only_texts, bm25_only_ids = [], []
        for rank, h in enumerate(bm25_hits):
            bm25_rank[h["id"]] = rank
            if h["id"] not in hits_by_id:
                bm25_only_ids.append(h["id"])
                bm25_only_texts.append(h["text"])
                hits_by_id[h["id"]] = {
                    "text": h["text"],
                    "source": h["meta"].get("source"),
                    "page": h["meta"].get("page"),
                    "distance": None,
                    "score": 0.0,  # placeholder, replaced below once embedded
                    "ocr_confidence": _meta_ocr_confidence(h["meta"]),
                }
        # a BM25-only hit has no cosine score yet -- compute one so it's
        # comparable to embedding-sourced hits wherever "score" is read
        # downstream (ambiguity detection, avg-confidence logging), instead
        # of leaving a placeholder that would always look like the weakest
        # possible match.
        if bm25_only_texts:
            embeddings = embedder.encode(bm25_only_texts, show_progress_bar=False)
            for hid, emb in zip(bm25_only_ids, embeddings):
                sim = _cosine_sim(query_embedding[0], emb)
                hits_by_id[hid]["score"] = sim
                hits_by_id[hid]["distance"] = 1 - sim

    if bm25_rank:
        rrf_scores = {
            hid: 1.0 / (config.RRF_K + embedding_rank.get(hid, 10**6) + 1)
            + 1.0 / (config.RRF_K + bm25_rank.get(hid, 10**6) + 1)
            for hid in hits_by_id
        }
        ordered_ids = sorted(hits_by_id.keys(), key=lambda hid: rrf_scores[hid], reverse=True)
    else:
        ordered_ids = sorted(hits_by_id.keys(), key=lambda hid: embedding_rank.get(hid, 10**6))

    ordered_ids = _dedupe_repeated_text(ordered_ids, hits_by_id)
    hits = [hits_by_id[hid] for hid in ordered_ids[:candidate_k]]

    if config.RERANK_ENABLED:
        hits = rerank(question, hits, k)
    else:
        hits = hits[:k]

    boost = _exact_identifier_boost(question, source_filter)
    if boost and not any(h["text"] == boost["text"] for h in hits):
        boost.setdefault("ocr_confidence", None)
        hits = _apply_boost(hits, [boost], k)

    own_doc_boosts = [
        b for b in _own_document_code_boost(source_filter)
        if not any(h["text"] == b["text"] for h in hits)
    ]
    if own_doc_boosts:
        for b in own_doc_boosts:
            b.setdefault("ocr_confidence", None)
        hits = _apply_boost(hits, own_doc_boosts, k)

    heading_boosts = [
        b for b in _heading_quote_boost(question, source_filter, k)
        if not any(h["text"] == b["text"] for h in hits)
    ]
    if heading_boosts:
        hits = _apply_boost(hits, heading_boosts, k)

    return hits


# ---------------------------------------------------------------------------
# Ambiguity detection
# ---------------------------------------------------------------------------

# Public (no leading underscore) -- build_eval_questions.py imports this
# directly so its bare-identifier eval category matches the exact same
# definition detect_ambiguity() uses, instead of a second regex drifting
# out of sync with it.
GENERIC_IDENTIFIER_PATTERN = re.compile(
    r"\b(pasal|lampiran|bab)\s+([A-Za-z]?\d+)", re.IGNORECASE
)


class AmbiguityResult:
    def __init__(self, candidates: list[str]):
        self.candidates = candidates


def detect_ambiguity(question: str, named_sources: list[str]) -> Optional[AmbiguityResult]:
    """Bare generic identifiers (e.g. "Pasal 8") with no named document,
    where that exact identifier appears as a heading in 2+ real documents,
    should trigger clarification instead of silently picking one. Bypassed
    whenever a document is actually named in the question.

    Structural (heading-count), not score-based -- confirmed live this
    matters (real eval false positive at 1k-corpus scale, "Apa isi Lampiran
    12?", which genuinely appears as a heading in 15+ documents). A
    reranked-score gap (the previous approach, and still what
    generate_answer()'s own competing-sources logic uses) conflates "one
    document has richer retrievable content under this heading" with "the
    identifier is unambiguous": most of those 15 documents' Lampiran-12
    sections are a single-line table-of-contents entry with little text,
    so only the one document with substantive content scored high enough
    to separate cleanly -- a wide score gap despite 14 other equally valid
    candidates. Heading presence doesn't have that bias. Reuses the exact
    same (identifier -> sources) grouping build_eval_questions.py's
    build_ambiguous_identifier() already uses to build ground truth, so
    detection and ground truth agree by construction.
    """
    if named_sources:
        return None
    match = GENERIC_IDENTIFIER_PATTERN.search(question)
    if not match:
        return None

    identifier = f"{match.group(1).lower()} {match.group(2)}"
    grouped = database.get_all_headings_grouped_by_source()
    matching_sources: set[str] = set()
    for source, headings in grouped.items():
        for h in headings:
            m = GENERIC_IDENTIFIER_PATTERN.search(h["text"])
            if m and f"{m.group(1).lower()} {m.group(2)}" == identifier:
                matching_sources.add(source)
                break

    if len(matching_sources) < 2:
        return None
    return AmbiguityResult(candidates=sorted(matching_sources))


# ---------------------------------------------------------------------------
# Scoped heading lookups
# ---------------------------------------------------------------------------

def get_headings_for(sources: list[str]) -> list[dict]:
    """Never call database.get_all_headings_grouped_by_source() for a
    per-question lookup -- always scope to the relevant document(s)
    (known-bug #2)."""
    return database.get_headings_for_sources(sources)


def _label_from_page1_chunks(chunks: list[tuple[dict, str]]) -> str:
    """Pure: given a document's own page-1 (metadata, text) chunk pairs,
    pick its real category label -- the first non-empty line of the
    lowest-chunk_index text, skipping clearly-too-short noise lines.
    Shared by get_document_type_label() and the batched
    get_document_type_labels()."""
    if not chunks:
        return "Tidak diketahui"
    ordered = sorted(chunks, key=lambda pair: pair[0].get("chunk_index", 0))
    lines = [l.strip() for l in ordered[0][1].splitlines() if l.strip()]
    if not lines:
        return "Tidak diketahui"
    # A cover page's very first OCR'd line is sometimes 1-2 characters of
    # logo/glyph noise (confirmed live on a real scanned cover) rather than
    # the real category text a couple of lines down -- skip clearly-too-
    # short lines first, but still return something rather than nothing if
    # every line on the page happens to be short.
    for line in lines:
        if len(line) >= 5:
            return line
    return lines[0]


def get_document_type_label(source: str) -> str:
    """A document's own real category label (e.g. "TATA KERJA ORGANISASI",
    "TATA KERJA INDIVIDU", "TATA KERJA PENGGUNAAN ALAT", "PEDOMAN") is
    reliably the first non-empty line of its own page 1 -- confirmed live
    across every real document type seen in this corpus (SOPs, individual
    procedures, equipment manuals, contracts, letters). Reading it directly
    from the document's own content replaces guessing from the filename
    (api.py's old _infer_doc_type() called anything not starting with
    "kontrak" a "sop", which was simply wrong for every other real
    document type/naming convention actually in this corpus). Returns
    whatever the document's own first line says, verbatim -- not
    classified into a fixed category set, since a hardcoded category enum
    would just be the same kind of corpus-specific guess in a different
    form.

    For more than a handful of sources, use get_document_type_labels()
    instead -- calling this once per document does one Chroma query per
    call, which doesn't scale (see that function's docstring)."""
    collection = get_collection()
    fetched = collection.get(
        where={"$and": [{"source": source}, {"page": 1}]}, include=["documents", "metadatas"]
    )
    chunks = list(zip(fetched.get("metadatas", []), fetched.get("documents", [])))
    return _label_from_page1_chunks(chunks)


def get_document_type_labels(sources: list[str]) -> dict[str, str]:
    """Batched version of get_document_type_label() -- ONE Chroma query for
    every document's page-1 chunks, grouped by source in Python, instead of
    one query per document. Confirmed live: api.py's /documents listing
    called the per-document version once per document (over a thousand documents in
    this corpus), executed serially and NOT through run_on_chroma_thread
    (the single-thread access pattern every other Chroma-touching endpoint
    in this app uses) -- this alone made /documents take minutes, and
    combined with concurrent access from other endpoints, could hang the
    whole API server. Callers should still route this through
    run_on_chroma_thread for the same reason every other Chroma access
    does; batching alone fixes the O(n) query count, not the thread-safety
    contract."""
    if not sources:
        return {}
    collection = get_collection()
    fetched = collection.get(
        where={"$and": [{"source": {"$in": sources}}, {"page": 1}]},
        include=["documents", "metadatas"],
    )
    by_source: dict[str, list[tuple[dict, str]]] = {}
    for meta, doc in zip(fetched.get("metadatas", []), fetched.get("documents", [])):
        by_source.setdefault(meta["source"], []).append((meta, doc))
    return {source: _label_from_page1_chunks(by_source.get(source, [])) for source in sources}


def get_full_document_text(source: str) -> str:
    """Reconstructs (approximately) the full document from its stored
    chunks, ordered by (page, chunk_index) -- used by the full-document
    fallback for small documents. Not a byte-perfect reconstruction (the
    sliding-window fallback in ingestion.py's chunker can leave a little
    overlap duplication), but that's harmless padding for a context-fit
    check and a one-shot full-text prompt, not something read as ground
    truth elsewhere."""
    collection = get_collection()
    fetched = collection.get(where={"source": source}, include=["documents", "metadatas"])
    docs = fetched.get("documents", [])
    metas = fetched.get("metadatas", [])
    ordered = sorted(
        zip(metas, docs), key=lambda pair: (pair[0].get("page", 0), pair[0].get("chunk_index", 0))
    )
    return "\n\n".join(doc for _, doc in ordered)


# ---------------------------------------------------------------------------
# Enumerate-intent routing
# ---------------------------------------------------------------------------
#
# "list all Pasal in kontrak 102" answered through normal similarity
# retrieval returned only 4 of the real ~25 -- top-k similarity search was
# never designed to guarantee completeness, it returns the *most relevant*
# chunks, not *all* of them. "Sebutkan semua"/"list all" phrasing already
# routed around retrieval via broad_category_listing() below, but a
# differently-phrased version of the same request ("apa saja pasal yang
# ada di kontrak 102") used none of the recognized trigger words and fell
# straight through to the incomplete top-k path -- confirmed live: the
# exact same question worded as "list all Pasal..." returned all 22 real
# headings, while "apa saja pasal..." returned None from this function and
# silently fell back to a 4-of-22 answer. Expanded triggers below close
# that specific gap; the deeper fix is that ANY of these trigger phrases
# combined with a real structural-element word must short-circuit before
# retrieval ever runs, which generate_answer() already does by checking
# this function first.

_ENUMERATE_TRIGGERS = (
    "list all", "daftar", "sebutkan semua", "apa saja", "berapa banyak", "semua", "ada berapa",
)


def _category_candidates_from_headings(headings: list[dict]) -> set[str]:
    """Category words ("Pasal", "Lampiran", ...) are pulled from the real
    headings actually stored for the document(s) in play, not a fixed list
    -- same principle as the identifier-boost and shorthand-matching
    vocabularies elsewhere in this module. A heading's first word, if
    alphabetic, is a category-word candidate; bare numeric headings (a
    contract's isolated ayat marker line, e.g. "11.1") contribute nothing
    here, which is correct -- "Ayat" isn't a listable category the way
    "Pasal"/"Lampiran" are, it's a sub-item within one."""
    # A single-letter lead ("B.", "C.", "E.") is this corpus's lettered
    # sub-item marker (e.g. "E. SESUATU"), not a real category word --
    # confirmed live it broke resolve_broad_category() below: its
    # substring `.find()` matches a bare "e" inside "Sebutkan" at position
    # 1, beating the real "lampiran"/"bab" candidate that appears later in
    # the question, so the feature silently returned the wrong category
    # for most real "list all X" questions. A real category word (bab,
    # lampiran, pasal...) is always >=3 letters in this corpus.
    words = set()
    for h in headings:
        first = h["text"].strip().split()[0] if h["text"].strip() else ""
        first = re.sub(r"[^A-Za-z]", "", first).lower()
        if len(first) >= 3:
            words.add(first)
    return words


def resolve_broad_category(question: str, candidates: set[str]) -> Optional[str]:
    """Pick the category word by the order it appears in the question, not
    unordered set iteration (known-bug #7). Word-boundary matching, not
    substring -- confirmed live a bare "e" candidate matched inside
    "Sebutkan" at position 1 with plain .find(), beating the real category
    word that only appears later in the question."""
    lower = question.lower()
    best_pos = None
    best_word = None
    for word in candidates:
        m = re.search(rf"\b{re.escape(word)}\b", lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_word = word
    return best_word


def _natural_sort_key(text: str):
    """Sorts "Pasal 2" before "Pasal 10" -- plain lexicographic sort put
    "Pasal 10" before "Pasal 2", which reads as broken/out-of-order to
    anyone who knows the document."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def broad_category_listing(question: str, named_sources: list[str]) -> Optional[dict]:
    """For questions like "list all lampiran/pasal sections" or "apa saja
    pasal yang ada". Resolves the category from real stored headings,
    requires >=2 distinct heading texts to qualify (filters out
    boilerplate/running-header artifacts), and formats the answer directly
    from headings with zero LLM call -- so the count can't be dropped by
    retrieval depth or hallucinated by the model."""
    is_enumerate = any(w in question.lower() for w in _ENUMERATE_TRIGGERS)
    if not is_enumerate:
        return None

    sources = named_sources or list_indexed_sources()
    headings = get_headings_for(sources)

    category = resolve_broad_category(question, _category_candidates_from_headings(headings))
    if not category:
        return None

    matching = [h for h in headings if h["text"].lower().startswith(category)]
    distinct_texts = sorted(set(h["text"] for h in matching), key=_natural_sort_key)

    if len(distinct_texts) < config.HEADING_MIN_CATEGORY_HEADINGS:
        return None

    return {"category": category, "headings": distinct_texts, "sources": sources}


# ---------------------------------------------------------------------------
# Document-type-level enumerate ("berapa banyak SOP yang ada", "sebutkan
# semua kontrak yang ada") -- a different kind of completeness question:
# not "list every heading of type X within a document", but "list/count
# every indexed FILE of type X". Handled separately from
# broad_category_listing() because the unit being counted is a document,
# not a heading.
# ---------------------------------------------------------------------------

def document_type_listing(question: str, named_sources: list[str]) -> Optional[dict]:
    # a specific document was already named ("kontrak 102") -- that means
    # the question is about content *within* that one document, not a
    # count/list of all documents of a type, even though "kontrak" appears
    # in both phrasings.
    if named_sources:
        return None
    is_enumerate = any(w in question.lower() for w in _ENUMERATE_TRIGGERS)
    if not is_enumerate:
        return None

    lower = question.lower()
    doc_type = None
    best_pos = None
    for word, pos in (("sop", lower.find("sop")), ("kontrak", lower.find("kontrak"))):
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
            doc_type = word
    if doc_type is None:
        return None

    sources = sorted(
        (s for s in list_indexed_sources() if s.lower().startswith(doc_type)),
        key=_natural_sort_key,
    )
    if not sources:
        return None

    return {"doc_type": doc_type, "sources": sources}
