"""Embedding-based retrieval, scoped heading lookups, named-document targeting, ambiguity
detection, and deterministic broad-category listing.
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

from app import config, database, lang

_COLLECTION_NAME = "documents"

_embedder: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None
_reranker: Optional[CrossEncoder] = None
_reranker_unavailable = False

_sources_cache = database.TTLCache(config.FILENAME_CACHE_TTL_SECONDS)
_bm25_index_cache: dict = {}
_vocab_cache: dict = {}

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
    """Lazily loads the cross-encoder."""
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
    """Cross-encoder reranking of the wider embedding-candidate set fetched by retrieve()."""
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
    """Short-TTL cached list of indexed documents, so one request does not recompute it several
    times.
    """
    return _sources_cache.get_or_set(database.list_document_sources)


def invalidate_sources_cache() -> None:
    _sources_cache.invalidate()
    _bm25_index_cache.clear()  # BM25 indexes are stale the moment the corpus changes
    _vocab_cache.clear()


def _contract_type_words(sources: list[str]) -> set[str]:
    filename_re = re.compile(
        rf"^{lang.get().CONTRACT_FILENAME_WORD}[_\s]+([a-zA-Z]+)[_\s]+\d", re.IGNORECASE
    )
    words = set()
    for source in sources:
        m = filename_re.match(source)
        if m:
            words.add(m.group(1).lower())
    return words


def _build_shorthand_patterns(sources: list[str]) -> list[re.Pattern]:
    L = lang.get()
    connector = rf"(?:(?:{L.NUMBER_WORDS})\s+)?"
    type_words = _contract_type_words(sources)
    type_group = ""
    if type_words:
        type_group = r"(?:(?:" + "|".join(re.escape(w) for w in sorted(type_words)) + r")\s+)?"
    return [
        re.compile(rf"\b({L.CONTRACT_WORDS})\s+" + type_group + connector + r"(\d+)", re.IGNORECASE),
        re.compile(r"\b(sop)\s+" + connector + r"(\d+)", re.IGNORECASE),
    ]


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_for_fuzzy_match(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


def named_target_sources(question: str) -> list[str]:
    """Indexed filenames referenced in the question, in the order they appear. The first-named
    document wins.
    """
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


_IDENTIFIER_EXTRACT_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:[ \t-]+[A-Z][a-zA-Z]*)*)[ \t]+([A-Za-z]?\d+(?:[.\-]\d+)?)\b"
)


def _exact_identifier_boost(question: str, source_filter: Optional[list[str]]) -> Optional[dict]:
    """Add the chunk whose heading carries the exact identifier named in the question (for
    example a clause number), because embeddings alone rank such literals poorly.
    """
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
    """Pages of the heading a question quotes by name, matched against the document's real
    headings.
    """
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


_DOC_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Z])[\s\-–]*(\d+)[\s\-–/]+([A-Za-z0-9]+)[\s\-–/]+"
    r"(\d{4})[\s\-–]*S[\s]*(\d+)(?![A-Za-z0-9])"
)


def _source_document_code(source: str) -> Optional[str]:
    match = _DOC_CODE_RE.search(source)
    if not match:
        return None
    return _normalize_for_fuzzy_match(match.group(0))


def _own_document_code_boost(source_filter: Optional[list[str]]) -> list[dict]:
    """When a question is scoped to exactly one document, make sure a chunk containing that
    document's OWN full document number is present in the retrieved set.
    """
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




def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_cache_key(source_filter: Optional[list[str]]):
    return tuple(sorted(source_filter)) if source_filter else None


def _get_bm25_index(source_filter: Optional[list[str]]):
    """One BM25 index per distinct source_filter scope, invalidated whenever the corpus changes
    (see invalidate_sources_cache()).
    """
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
    """Keyword (BM25) retrieval pass that runs alongside the embedding pass."""
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


# Words that carry no topic, in English and Indonesian.
_STOPWORDS = frozenset("""
what which who whom whose when where why how does did the this that these those with from into for and are was were
been being have has had can could should would will might must shall not any all each some such than then there their
they them you your our its about according under over also only other more most less very much many
apa siapa kapan dimana mengapa bagaimana berapa yang dan atau dengan dari untuk pada dalam adalah itu ini oleh para
telah akan dapat harus sebagai tentang menurut apakah bolehkah
""".split())


def _stem(word: str) -> str:
    for suffix in ("nya", "ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def corpus_vocabulary() -> tuple[set[str], set[str]]:
    """Every word in the indexed documents, and the same words with endings removed. Cached until the corpus changes."""
    if "words" not in _vocab_cache:
        entry = _get_bm25_index(None)
        words = {w for doc in entry[2] for w in _tokenize(doc)} if entry else set()
        _vocab_cache["words"] = (words, {_stem(w) for w in words})
    return _vocab_cache["words"]


def _is_specific(word: str) -> bool:
    """An acronym ("GDPR"), or a word the embedding model has to cut into three or more pieces ("Kerberos").
    Everyday words such as "minimum" or "length" are one piece, and a document may simply word them differently.
    """
    return word.isupper() or len(get_embedder().tokenizer.tokenize(word.lower())) >= 3


def unknown_terms(question: str) -> list[str]:
    """Specific words in the question that appear nowhere in the indexed documents (for example "Kerberos")."""
    words, stems = corpus_vocabulary()
    if not words:
        return []
    found = re.findall(r"[^\W\d_]{4,}|\b[A-Z][A-Z0-9]{2,}\b", question)
    return [
        w for w in found
        if (low := w.lower()) not in _STOPWORDS and low not in words and _stem(low) not in stems and _is_specific(w)
    ]


def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _dedupe_repeated_text(ordered_ids: list[str], hits_by_id: dict[str, dict]) -> list[str]:
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
    """Prepend forced boost items ahead of the normally-ranked hits, respecting the k budget on
    both the boost itself and the combined result.
    """
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
        try:
            filtered_ids = collection.get(where=where, include=[]).get("ids", [])
        except Exception:
            filtered_ids = None
        if filtered_ids is not None:
            candidate_k = max(1, min(candidate_k, len(filtered_ids)))
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


def _identifier_pattern() -> re.Pattern:
    words = "|".join(w for mod in lang.all_modules() for w in mod.IDENTIFIER_WORDS)
    return re.compile(rf"\b({words})\s+([A-Za-z]?\d+)", re.IGNORECASE)


class AmbiguityResult:
    def __init__(self, candidates: list[str]):
        self.candidates = candidates


def detect_ambiguity(question: str, named_sources: list[str]) -> Optional[AmbiguityResult]:
    """Detect a bare identifier (for example 'Appendix 1') that exists in several documents and
    return those documents, or None.
    """
    if named_sources:
        return None
    identifier_pattern = _identifier_pattern()
    match = identifier_pattern.search(question)
    if not match:
        return None

    identifier = f"{match.group(1).lower()} {match.group(2)}"
    grouped = database.get_all_headings_grouped_by_source()
    matching_sources: set[str] = set()
    for source, headings in grouped.items():
        for h in headings:
            m = identifier_pattern.search(h["text"])
            if m and f"{m.group(1).lower()} {m.group(2)}" == identifier:
                matching_sources.add(source)
                break

    if len(matching_sources) < 2:
        return None
    return AmbiguityResult(candidates=sorted(matching_sources))


def get_headings_for(sources: list[str]) -> list[dict]:
    """Headings for the given documents only. Always scope per question, never read the whole
    corpus.
    """
    return database.get_headings_for_sources(sources)


def _label_from_page1_chunks(chunks: list[tuple[dict, str]]) -> str:
    """Pure: given a document's own page-1 (metadata, text) chunk pairs, pick its real category
    label -- the first non-empty line of the lowest-chunk_index text, skipping clearly-too-
    short noise lines.
    """
    if not chunks:
        return lang.get().UNKNOWN_TYPE
    ordered = sorted(chunks, key=lambda pair: pair[0].get("chunk_index", 0))
    lines = [l.strip() for l in ordered[0][1].splitlines() if l.strip()]
    if not lines:
        return lang.get().UNKNOWN_TYPE
    for line in lines:
        if len(line) >= 5:
            return line
    return lines[0]


def get_document_type_label(source: str) -> str:
    """A document's own category label, taken from its first heading."""
    collection = get_collection()
    fetched = collection.get(
        where={"$and": [{"source": source}, {"page": 1}]}, include=["documents", "metadatas"]
    )
    chunks = list(zip(fetched.get("metadatas", []), fetched.get("documents", [])))
    return _label_from_page1_chunks(chunks)


def get_document_type_labels(sources: list[str]) -> dict[str, str]:
    """Batched version of get_document_type_label() -- ONE Chroma query for every document's
    page-1 chunks, grouped by source in Python, instead of one query per document.
    """
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
    """Reconstructs (approximately) the full document from its stored chunks, ordered by (page,
    chunk_index) -- used by the full-document fallback for small documents.
    """
    collection = get_collection()
    fetched = collection.get(where={"source": source}, include=["documents", "metadatas"])
    docs = fetched.get("documents", [])
    metas = fetched.get("metadatas", [])
    ordered = sorted(
        zip(metas, docs), key=lambda pair: (pair[0].get("page", 0), pair[0].get("chunk_index", 0))
    )
    return "\n\n".join(doc for _, doc in ordered)


def _category_candidates_from_headings(headings: list[dict]) -> set[str]:
    """Category words ("Article", "Appendix", ...) are pulled from the real headings actually
    stored for the document(s) in play, not a fixed list.
    """
    words = set()
    for h in headings:
        first = h["text"].strip().split()[0] if h["text"].strip() else ""
        first = re.sub(r"[^A-Za-z]", "", first).lower()
        if len(first) >= 3:
            words.add(first)
    return words


def resolve_broad_category(question: str, candidates: set[str]) -> Optional[str]:
    """Pick the category word by the order it appears in the question, not by set iteration
    order.
    """
    lower = question.lower()
    best_pos = None
    best_word = None
    for word in candidates:
        plural = rf"{re.escape(word[:-2])}(?:ix|ices)" if word.endswith("ix") else rf"{re.escape(word)}s?"
        m = re.search(rf"\b{plural}\b", lower)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_word = word
    return best_word


def _natural_sort_key(text: str):
    """Sorts "Article 2" before "Article 10" -- plain lexicographic sort put
    "Article 10" before "Article 2", which reads as broken/out-of-order to
    anyone who knows the document."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def broad_category_listing(question: str, named_sources: list[str]) -> Optional[dict]:
    """For questions like "list all appendix sections"."""
    is_enumerate = any(w in question.lower() for w in lang.get().ENUMERATE_TRIGGERS)
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


def document_type_listing(question: str, named_sources: list[str]) -> Optional[dict]:
    if named_sources:
        return None
    is_enumerate = any(w in question.lower() for w in lang.get().ENUMERATE_TRIGGERS)
    if not is_enumerate:
        return None

    lower = question.lower()
    doc_type = None
    best_pos = None
    for word in ("sop", lang.get().CONTRACT_FILENAME_WORD):
        pos = lower.find(word)
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
