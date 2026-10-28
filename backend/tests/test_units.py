"""Unit tests for pure/isolated functions -- no Chroma, sqlite, or Ollama.
Fast, deterministic, safe to run any time (including while ingestion or the
API server is up, since nothing here touches shared state).

Usage (from backend/): python -m pytest -v
"""
from __future__ import annotations

from app import generation, ingestion, retrieval


# ---------------------------------------------------------------------------
# generation.split_questions
# ---------------------------------------------------------------------------

def test_split_single_question_unchanged():
    assert generation.split_questions("Apa isi Pasal 8?") == ["Apa isi Pasal 8?"]


def test_split_two_questions_on_question_mark_conjunction():
    result = generation.split_questions("Apa isi Pasal 8? dan apa isi Pasal 9?")
    assert len(result) == 2
    assert "Pasal 8" in result[0]
    assert "Pasal 9" in result[1]


def test_split_conjunction_without_trailing_mark_before_each_clause():
    # single trailing "?" but a real second question hiding after "dan"
    result = generation.split_questions(
        "apa isi pasal 8 di kontrak ini dan apa isi lampiran A?"
    )
    assert len(result) == 2


def test_split_does_not_split_ordinary_dan_within_one_question():
    result = generation.split_questions("Bagaimana cara melakukannya dengan cepat dan benar?")
    assert len(result) == 1


def test_split_trailing_style_instruction_folds_into_previous_question():
    # "tolong jawab singkat" has no topic of its own -- must not become a
    # separate, empty-retrieval sub-question
    result = generation.split_questions("Apa isi Pasal 8? Tolong jawab singkat.")
    assert len(result) == 1
    assert "Pasal 8" in result[0]


def test_split_never_returns_empty_list_for_nonempty_input():
    result = generation.split_questions("???")
    assert result == [] or all(result)  # never a list containing an empty string


def test_split_empty_input_returns_empty_list():
    assert generation.split_questions("") == []
    assert generation.split_questions("   ") == []


# ---------------------------------------------------------------------------
# generation.unverified_identifiers / heading_lead_words
# ---------------------------------------------------------------------------

_HEADINGS = [
    {"page": 1, "text": "Pasal 8 Pemutusan Kontrak"},
    {"page": 2, "text": "Lampiran C2 Asuransi"},
    {"page": 3, "text": "1.2"},  # bare identifier, no leading word
]


def test_unverified_identifiers_catches_fabricated_number():
    # Pasal 99 does not exist in the heading index above
    result = generation.unverified_identifiers("Menurut Pasal 99, hal ini diatur.", _HEADINGS)
    assert any("99" in item for item in result)


def test_unverified_identifiers_accepts_grounded_reference():
    result = generation.unverified_identifiers("Sesuai Pasal 8, kontrak dapat diputus.", _HEADINGS)
    assert not any("Pasal 8" in item or item.lower() == "pasal 8" for item in result)


def test_unverified_identifiers_accepts_bare_ayat_number():
    # "1.2" is stored with no leading word -- heading_lead_words() keys on
    # the number alone for exactly this case
    result = generation.unverified_identifiers("Ayat 1.2 mengatur hal tersebut.", _HEADINGS)
    assert not any("1.2" in item for item in result)


def test_unverified_identifiers_ignores_currency_value():
    # "Sebesar 74" must never be mistaken for a section identifier
    result = generation.unverified_identifiers("Nilainya Sebesar 74 juta rupiah.", _HEADINGS)
    assert not any("Sebesar" in item for item in result)


def test_unverified_identifiers_ignores_calendar_date():
    result = generation.unverified_identifiers("Berlaku mulai tanggal 1 November 2024.", _HEADINGS)
    assert not any("November" in item for item in result)


def test_unverified_identifiers_no_claims_returns_empty():
    assert generation.unverified_identifiers("Tidak ada informasi spesifik di sini.", _HEADINGS) == []


def test_unverified_identifiers_page_citation_not_flagged_as_heading():
    text = "Menurut Halaman 72 dan Halaman 24, ini berlaku."
    assert generation.unverified_identifiers(text, _HEADINGS, {72, 24}) == []


def test_unverified_identifiers_page_not_in_retrieved_pages_is_flagged():
    result = generation.unverified_identifiers("Lihat Halaman 99.", _HEADINGS, {72, 24})
    assert len(result) == 1 and result[0].endswith("Halaman 99")


def test_unverified_identifiers_unknown_pages_skips_page_check():
    assert generation.unverified_identifiers("Lihat Halaman 99.", _HEADINGS, None) == []


def test_unverified_identifiers_deduplicates_repeated_claim():
    result = generation.unverified_identifiers("Pasal 99 dan lagi Pasal 99.", _HEADINGS)
    assert result == ["Pasal 99"]


def test_build_prompt_single_document_named_once_with_page_only_labels():
    hits = [{"source": "A.pdf", "page": 3, "text": "isi satu"}, {"source": "A.pdf", "page": 5, "text": "isi dua"}]
    prompt = generation.build_prompt("tanya?", hits)
    assert prompt.count("A.pdf") == 1
    assert "[Halaman 3]" in prompt and "[Halaman 5]" in prompt


def test_build_prompt_multiple_documents_keep_source_in_each_label():
    hits = [{"source": "A.pdf", "page": 3, "text": "x"}, {"source": "B.pdf", "page": 1, "text": "y"}]
    prompt = generation.build_prompt("tanya?", hits)
    assert "[Sumber: A.pdf, Halaman 3]" in prompt and "[Sumber: B.pdf, Halaman 1]" in prompt


# ---------------------------------------------------------------------------
# generation.looks_like_refusal / looks_like_hedging
# ---------------------------------------------------------------------------

def test_looks_like_refusal_detects_indonesian_phrase():
    assert generation.looks_like_refusal("Maaf, informasi tersebut tidak ditemukan dalam dokumen.")


def test_looks_like_refusal_false_for_normal_answer():
    assert not generation.looks_like_refusal("Pasal 8 mengatur tentang pemutusan kontrak.")


def test_looks_like_refusal_false_for_clinical_not_found_phrase():
    # Confirmed live in the real 100-question eval: a genuinely correct,
    # detailed medical-fitness answer (real content, all 7 real health
    # categories P1-P7) got wrongly flagged as a refusal because it
    # legitimately contains "Tidak ditemukan kelainan medis" (no medical
    # abnormality found -- a real clinical category name). Semantic
    # similarity to the system's own actual refusal framing is what
    # correctly tells these apart (verified: 0.207 here vs 0.68-0.73 for
    # genuine refusals).
    answer = (
        "Bagi kategori kesehatan tersebut, terdapat tujuh kategori:\n"
        "**Tidak ditemukan kelainan medis (P1)** -- Tidak ditemukan kelainan "
        "medis dan SKJ rendah."
    )
    assert not generation.looks_like_refusal(answer)


def test_looks_like_hedging_detects_uncertainty_phrase():
    assert generation.looks_like_hedging("Kemungkinan hal ini berlaku, namun perlu diverifikasi.")


# ---------------------------------------------------------------------------
# generation._answer_sentences / sentence_groundedness edge cases
# ---------------------------------------------------------------------------

def test_answer_sentences_strips_footer():
    answer = "Ini jawaban asli.\n\n---\nCatatan tambahan.\nDihasilkan otomatis -- verifikasi ulang."
    sentences = generation._answer_sentences(answer)
    assert any("jawaban asli" in s for s in sentences)
    assert not any("Dihasilkan otomatis" in s for s in sentences)


def test_sentence_groundedness_empty_answer_returns_none():
    result = generation.sentence_groundedness("", [{"text": "some passage"}])
    assert result["min"] is None
    assert result["per_sentence"] == []


def test_sentence_groundedness_no_hits_returns_none():
    result = generation.sentence_groundedness("Some answer text here.", [])
    assert result["min"] is None


# ---------------------------------------------------------------------------
# retrieval._category_candidates_from_headings / resolve_broad_category
# ---------------------------------------------------------------------------

def test_category_candidates_excludes_lettered_sub_items():
    # "E. SESUATU" is a lettered sub-item marker, not a real category word --
    # confirmed live this broke real "list all X" questions in production.
    headings = [
        {"text": "E. SESUATU"},
        {"text": "BAB I UMUM"},
        {"text": "BAB II KEBIJAKAN"},
    ]
    candidates = retrieval._category_candidates_from_headings(headings)
    assert candidates == {"bab"}


def test_resolve_broad_category_picks_real_word_not_letter_substring():
    # Before the fix, a bare "e" candidate matched inside "Sebutkan" at
    # position 1 and won over "bab", which only appears later in the
    # question -- resolve_broad_category returned "e" instead of "bab".
    question = "Sebutkan semua bab yang ada di dokumen X.pdf."
    assert retrieval.resolve_broad_category(question, {"bab", "e"}) == "bab"


# ---------------------------------------------------------------------------
# generation._passes_relevance_gate
# ---------------------------------------------------------------------------

def test_relevance_gate_passes_on_high_rerank_despite_low_raw_score():
    # Confirmed live: a genuine but short/generic question ("cara mengajukan
    # cuti") can score LOWER on raw embedding similarity than an off-topic
    # one ("resep rendang": 0.574) -- 0.514 for the real question. The
    # cross-encoder rerank score separates the two cleanly (off-topic
    # ceiling -0.63, genuine floor ~2.1), so a hit clearing EITHER signal
    # must count as relevant.
    hits = [{"score": 0.50, "rerank_score": 5.0}]
    assert generation._passes_relevance_gate(hits) is True


def test_relevance_gate_rejects_when_both_signals_are_low():
    hits = [{"score": 0.40, "rerank_score": -2.0}]
    assert generation._passes_relevance_gate(hits) is False


def test_relevance_gate_rejects_empty_hits():
    assert generation._passes_relevance_gate([]) is False


def test_relevance_gate_fails_closed_when_rerank_score_missing():
    # If a hit has no rerank_score (e.g. RERANK_ENABLED were ever False),
    # falling back to the raw cosine score would compare a 0-1 value
    # against RELEVANCE_MIN_RERANK_SCORE=0.0 -- a threshold calibrated for
    # the cross-encoder's unbounded logit scale -- making the OR condition
    # trivially true for almost any hit and silently defeating the gate.
    # Missing rerank score must contribute nothing, not a free pass.
    hits = [{"score": 0.30}]  # below RELEVANCE_MIN_SCORE, no rerank_score at all
    assert generation._passes_relevance_gate(hits) is False


# ---------------------------------------------------------------------------
# retrieval._dedupe_repeated_text
# ---------------------------------------------------------------------------

def test_dedupe_collapses_identical_text_within_same_source():
    # Confirmed live: a running page header repeated verbatim across a
    # multi-page attachment (e.g. "Lampiran 19..." on pages 91-100 of a
    # real document) can occupy most of top-k on a query that quotes the
    # heading, crowding out the one chunk on page 94 that has the actual
    # body content -- the LLM then honestly reports "not found" despite
    # the real answer being in the corpus, unretrieved.
    hits_by_id = {
        "a": {"source": "doc1.pdf", "text": "HEADER TEXT"},
        "b": {"source": "doc1.pdf", "text": "HEADER TEXT"},
        "c": {"source": "doc1.pdf", "text": "REAL BODY CONTENT"},
    }
    result = retrieval._dedupe_repeated_text(["a", "b", "c"], hits_by_id)
    assert result == ["a", "c"]


def test_find_quoted_heading_pages_matches_when_most_words_present():
    # Confirmed live: a question asking "what's in heading H" quotes H's
    # own words almost verbatim -- requiring most (not all, tolerate minor
    # rewording) of H's distinguishing words to appear in the question
    # avoids matching a short/generic heading by coincidence, the same
    # false-positive guard _exact_identifier_boost already uses.
    headings = [
        {"text": "Lampiran 19 Pedoman Resume Aktivitas Proses Konsultasi", "page": 94},
        {"text": "BAB II KEBIJAKAN", "page": 5},
    ]
    question = "Apa isi bagian 'Lampiran 19 Pedoman Resume Aktivitas Proses Konsultasi' pada dokumen X.pdf?"
    assert retrieval._find_quoted_heading_pages(question, headings) == [94]


def test_find_quoted_heading_pages_collects_all_pages_of_a_repeated_heading():
    # A running header repeated across a multi-page attachment (confirmed
    # live: "Lampiran 19..." on pages 91-100 of a real document) must
    # surface ALL of its pages, not just the first match -- the real body
    # content can be on any of them.
    headings = [
        {"text": "Lampiran 19 Resume Aktivitas Konsultasi", "page": 91},
        {"text": "Lampiran 19 Resume Aktivitas Konsultasi", "page": 94},
        {"text": "Lampiran 19 Resume Aktivitas Konsultasi", "page": 100},
        {"text": "Lampiran 20 Laporan Hasil Konsultasi", "page": 101},
    ]
    question = "Apa isi bagian 'Lampiran 19 Resume Aktivitas Konsultasi' pada dokumen X.pdf?"
    assert retrieval._find_quoted_heading_pages(question, headings) == [91, 94, 100]


def test_find_quoted_heading_pages_returns_empty_when_no_real_overlap():
    headings = [{"text": "Lampiran 19 Pedoman Resume Aktivitas Proses Konsultasi", "page": 94}]
    question = "Apa itu manajemen risiko?"
    assert retrieval._find_quoted_heading_pages(question, headings) == []


def test_find_quoted_heading_pages_ignores_coincidental_word_overlap_without_quotes():
    # Confirmed live (independent code review + real corpus check): a bare,
    # generic 2-word heading like "RUANG LINGKUP" or "TATA CARA" -- 292
    # distinct examples exist in the real corpus, including "DAFTAR ISI" --
    # matched ANY question mentioning those words in passing, with no
    # quoting intent at all, and hijacked retrieval to an arbitrary page.
    # The real use case always quotes the heading ("Apa isi bagian 'X' pada
    # dokumen Y") -- requiring an actual quoted span, not just word overlap
    # against the whole question, is what distinguishes genuine "asking
    # about this specific heading" from a question that merely uses common
    # words.
    headings = [{"text": "RUANG LINGKUP", "page": 3}]
    question = "Apa ruang lingkup dari pedoman ini?"
    assert retrieval._find_quoted_heading_pages(question, headings) == []


def test_find_quoted_heading_pages_matches_inside_quotes_even_if_short():
    headings = [{"text": "RUANG LINGKUP", "page": 3}]
    question = "Apa isi bagian 'RUANG LINGKUP' pada dokumen ini?"
    assert retrieval._find_quoted_heading_pages(question, headings) == [3]


def test_dedupe_keeps_identical_text_across_different_sources():
    # Two unrelated documents sharing boilerplate text (e.g. a title-block
    # template) is not the same problem -- only same-source repeats should
    # collapse.
    hits_by_id = {
        "a": {"source": "doc1.pdf", "text": "SAME BOILERPLATE"},
        "b": {"source": "doc2.pdf", "text": "SAME BOILERPLATE"},
    }
    result = retrieval._dedupe_repeated_text(["a", "b"], hits_by_id)
    assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# ingestion._image_content_hash
# ---------------------------------------------------------------------------

def test_image_content_hash_is_stable_for_identical_bytes():
    assert ingestion._image_content_hash(b"same bytes") == ingestion._image_content_hash(b"same bytes")


def test_image_content_hash_differs_for_different_bytes():
    assert ingestion._image_content_hash(b"bytes a") != ingestion._image_content_hash(b"bytes b")


def test_image_content_hash_differs_when_cache_context_differs():
    # Confirmed live (independent code review): the cache key originally
    # hashed only the image bytes, with nothing about the VLM prompt or
    # model -- changing either in the future would silently keep serving
    # old cached descriptions with no signal anything was stale. Folding a
    # cache-busting context string (prompt + model name) into the key means
    # changing either one naturally invalidates old entries instead of
    # requiring a manual cache wipe.
    same_bytes = b"identical image bytes"
    assert ingestion._image_content_hash(same_bytes, "prompt v1|model a") != \
        ingestion._image_content_hash(same_bytes, "prompt v2|model a")


# ---------------------------------------------------------------------------
# retrieval._apply_boost
# ---------------------------------------------------------------------------

def test_apply_boost_caps_a_single_oversized_boost_to_k():
    # Confirmed live (independent code review): a boost list itself bigger
    # than k, combined with the old `hits[:max(0, k-len(boost))]` formula,
    # let the final result exceed the k budget entirely -- the boost list
    # was never capped down to k on its own.
    hits = [{"text": "normal"}]
    boost_items = [{"text": f"boost{i}"} for i in range(10)]
    result = retrieval._apply_boost(hits, boost_items, k=5)
    assert len(result) == 5


def test_apply_boost_caps_stacked_boosts_to_k():
    # Two boosts applied in sequence must never exceed k combined, even
    # though each one only ever sees its own boost_items list.
    hits = [{"text": "normal"}]
    first = retrieval._apply_boost(hits, [{"text": "a"}, {"text": "b"}], k=3)
    second = retrieval._apply_boost(first, [{"text": "c"}, {"text": "d"}], k=3)
    assert len(second) == 3


def test_apply_boost_keeps_normal_hits_when_boost_is_empty():
    hits = [{"text": "one"}, {"text": "two"}]
    assert retrieval._apply_boost(hits, [], k=5) == hits


# ---------------------------------------------------------------------------
# retrieval._label_from_page1_chunks / get_document_type_labels
# ---------------------------------------------------------------------------

def test_label_from_page1_chunks_picks_first_line_of_lowest_chunk_index():
    chunks = [
        ({"chunk_index": 1}, "SOME OTHER TEXT\nMore text"),
        ({"chunk_index": 0}, "TATA KERJA ORGANISASI\nRest of page"),
    ]
    assert retrieval._label_from_page1_chunks(chunks) == "TATA KERJA ORGANISASI"


def test_label_from_page1_chunks_skips_short_noise_lines():
    # Confirmed live: a scanned cover page's first OCR'd line can be 1-2
    # characters of logo/glyph noise, not the real category text.
    chunks = [({"chunk_index": 0}, "X\nPEDOMAN PENGELOLAAN RISIKO")]
    assert retrieval._label_from_page1_chunks(chunks) == "PEDOMAN PENGELOLAAN RISIKO"


def test_label_from_page1_chunks_returns_unknown_for_empty_chunks():
    assert retrieval._label_from_page1_chunks([]) == "Tidak diketahui"


def test_get_document_type_labels_returns_empty_dict_for_empty_sources():
    # No sources means no Chroma query at all -- this must short-circuit
    # before touching get_collection(), so it's testable without a live
    # collection.
    assert retrieval.get_document_type_labels([]) == {}
