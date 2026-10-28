"""English counterparts of the language-dependent unit tests, plus checks that both language modules agree."""
from __future__ import annotations

import pytest

from app import generation, lang, retrieval
from app.lang import ui


@pytest.fixture(autouse=True)
def english():
    lang.set_language("en")
    yield
    lang.set_language("id")


def test_modules_define_the_same_names():
    en = lang.set_language("en")
    id_ = lang.set_language("id")
    public = lambda m: {n for n in dir(m) if n.isupper()}
    assert public(en) == public(id_)


def test_message_templates_format_in_both_languages():
    for code in ("en", "id"):
        L = lang.set_language(code)
        L.CLARIFY.format(candidates="a.pdf, b.pdf")
        L.MULTI_SOURCE_NOTE.format(count=2, top="a.pdf")
        L.BROAD_LISTING.format(count=1, category="x", lines="- y")
        L.DOC_TYPE_LISTING.format(count=1, doc_type="x", lines="- y")
        L.UNVERIFIED_NOTE.format(items="Article 9")
        L.LOW_OCR_NOTE.format(confidence=42)
        L.RESUME_TEMPLATE.format(question="q", document="d")


def test_resume_template_round_trips_through_the_resume_pattern():
    for code in ("en", "id"):
        L = lang.set_language(code)
        text = L.RESUME_TEMPLATE.format(question="What does Appendix 1 say?", document="a.pdf")
        match = L.RESUME_RE.match(text)
        assert match and match.group(1) == "What does Appendix 1 say?" and match.group(2) == "a.pdf"


def test_split_two_questions():
    result = generation.split_questions("What is in Article 8? and what is in Article 9?")
    assert len(result) == 2
    assert "Article 8" in result[0] and "Article 9" in result[1]


def test_split_does_not_split_ordinary_and_within_one_question():
    assert len(generation.split_questions("How do I do it quickly and correctly?")) == 1


def test_split_style_instruction_folds_into_previous_question():
    result = generation.split_questions("What is in Article 8? Please be brief.")
    assert len(result) == 1 and "Article 8" in result[0]


def test_unverified_identifiers_catches_fabricated_number():
    headings = [{"text": "Article 5 Leave"}]
    flagged = generation.unverified_identifiers("The rule is in Article 99.", headings)
    assert len(flagged) == 1 and flagged[0].endswith("Article 99")


def test_unverified_identifiers_accepts_grounded_reference():
    headings = [{"text": "Article 5 Leave"}]
    assert generation.unverified_identifiers("See Article 5 for details.", headings) == []


def test_unverified_identifiers_ignores_amounts_and_dates():
    assert generation.unverified_identifiers("Pay is $30 per day, from March 2026.", []) == []
    assert generation.unverified_identifiers("The limit is Total 12 days.", []) == []


def test_unverified_identifiers_page_citation_checked_against_retrieved_pages():
    assert generation.unverified_identifiers("It is on Page 3.", [], {3}) == []
    assert generation.unverified_identifiers("It is on Page 9.", [], {3}) == ["Page 9"]


def test_build_prompt_is_english_with_page_labels():
    prompt = generation.build_prompt("What is the limit?", [{"source": "a.pdf", "page": 2, "text": "x"}])
    assert "[Page 2]" in prompt and "QUESTION: What is the limit?" in prompt
    assert "Halaman" not in prompt and "PERTANYAAN" not in prompt


def test_build_prompt_multiple_documents_keep_source_in_each_label():
    hits = [{"source": "a.pdf", "page": 1, "text": "x"}, {"source": "b.pdf", "page": 4, "text": "y"}]
    prompt = generation.build_prompt("q", hits)
    assert "[Source: a.pdf, Page 1]" in prompt and "[Source: b.pdf, Page 4]" in prompt


def test_looks_like_refusal_detects_english_phrase():
    assert generation.looks_like_refusal("Sorry, that information was not found in the available documents.")


def test_looks_like_refusal_false_for_normal_answer():
    assert not generation.looks_like_refusal("Annual leave is 12 days after one year of service.")


def test_looks_like_hedging():
    assert generation.looks_like_hedging("It seems the limit may be 12 days.")
    assert not generation.looks_like_hedging("The limit is 12 days.")


def test_identifier_pattern_knows_both_languages():
    pattern = retrieval._identifier_pattern()
    assert pattern.search("What does Appendix 1 say?").group(2) == "1"
    assert pattern.search("Apa isi Lampiran 1?").group(2) == "1"


def test_detect_picks_the_language_of_the_question():
    assert lang.detect("What does Appendix 1 say?") == "en"
    assert lang.detect("Berapa hari cuti tahunan karyawan?") == "id"
    assert lang.detect("Apa isi Lampiran 1 untuk dokumen a.pdf") == "id"
    assert lang.detect("12345") == "en"  # unclear: falls back to the default language


def test_use_switches_language_only_inside_the_block():
    assert lang.get().CODE == "en"
    with lang.use("id"):
        assert lang.get().CODE == "id"
        assert generation.build_prompt("q", [{"source": "a", "page": 1, "text": "x"}]).count("PERTANYAAN") == 1
    assert lang.get().CODE == "en"


def test_split_keeps_a_clause_that_refers_back_together():
    result = generation.split_questions("How long must a password be and how often must it be changed?")
    assert len(result) == 1


def test_broad_category_matches_plural_words():
    assert retrieval.resolve_broad_category("list all appendices", {"appendix", "article"}) == "appendix"
    assert retrieval.resolve_broad_category("list all articles", {"appendix", "article"}) == "article"


def test_resume_text_in_either_language_is_recognised():
    for mod in lang.all_modules():
        assert any(m.RESUME_RE.match(mod.RESUME_TEMPLATE.format(question="q", document="a.pdf")) for m in lang.all_modules())


def test_enumerate_triggers_are_english():
    assert any(t in "list all the appendices".lower() for t in lang.get().ENUMERATE_TRIGGERS)


def test_ui_strings_are_complete_english():
    assert all(v.strip() for v in ui.STRINGS.values())
    assert {"disclaimer", "composer_placeholder", "resume_template", "new_chat"} <= set(ui.STRINGS)
