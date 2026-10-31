"""The relevance gate's check for topic words that appear nowhere in the documents."""
import pytest

from app import config, generation, retrieval

WORDS = {"password", "passwords", "length", "memorized", "secret", "incident", "response", "data"}


@pytest.fixture(autouse=True)
def small_vocabulary(monkeypatch):
    monkeypatch.setattr(retrieval, "corpus_vocabulary", lambda: (WORDS, {retrieval._stem(w) for w in WORDS}))
    monkeypatch.setattr(retrieval, "_is_specific", lambda w: True)  # the real check is tested separately below


def test_unknown_terms_finds_a_word_the_documents_never_use():
    assert retrieval.unknown_terms("How does Kerberos authentication work?") == ["Kerberos", "authentication", "work"]


def test_unknown_terms_ignores_question_words_short_words_and_numbers():
    assert retrieval.unknown_terms("What is the password length for 2023 in the US?") == []


def test_unknown_terms_ignores_word_endings():
    # "memorizing" and "memorized" share a stem, and "passwords" matches "password"
    assert retrieval.unknown_terms("memorizing passwords") == []
    assert retrieval.unknown_terms("Apa itu datanya?") == []


def test_is_specific_tells_acronyms_and_rare_words_from_everyday_words(monkeypatch):
    monkeypatch.undo()  # use the real tokenizer
    assert retrieval._is_specific("GDPR") and retrieval._is_specific("Kerberos")
    assert not retrieval._is_specific("minimum") and not retrieval._is_specific("length")


def test_unknown_terms_is_empty_for_an_empty_index(monkeypatch):
    monkeypatch.setattr(retrieval, "corpus_vocabulary", lambda: (set(), set()))
    assert retrieval.unknown_terms("anything at all") == []


def test_gate_rejects_a_weak_match_when_a_topic_word_is_unknown():
    hits = [{"score": 0.70, "rerank_score": 1.5}]
    assert generation._passes_relevance_gate(hits) is True  # no question given, the old behaviour
    assert generation._passes_relevance_gate(hits, "What is the GDPR fine?") is False


def test_gate_lets_a_very_strong_match_override_an_unknown_word():
    hits = [{"score": 0.70, "rerank_score": config.UNKNOWN_TERM_RERANK_MIN + 0.5}]
    assert generation._passes_relevance_gate(hits, "What is the GDPR fine?") is True


def test_gate_is_unchanged_when_every_topic_word_is_known():
    hits = [{"score": 0.70, "rerank_score": 1.5}]
    assert generation._passes_relevance_gate(hits, "password length") is True


def test_gate_without_rerank_scores_does_not_apply_the_check():
    hits = [{"score": 0.70}]
    assert generation._passes_relevance_gate(hits, "What is the GDPR fine?") is True


def test_vocabulary_follows_the_index_when_documents_change(monkeypatch):
    monkeypatch.undo()  # use the real corpus_vocabulary, with a fake index underneath
    docs = ["a note about passwords"]
    monkeypatch.setattr(retrieval, "_get_bm25_index", lambda source_filter: (None, [], list(docs), []))
    retrieval.invalidate_sources_cache()
    assert "kerberos" not in retrieval.corpus_vocabulary()[0]

    docs.append("Kerberos tickets expire after ten hours")  # a new file was indexed
    retrieval.invalidate_sources_cache()  # sync_once does this whenever the corpus changes
    assert "kerberos" in retrieval.corpus_vocabulary()[0]

    docs.pop()  # and then removed again
    retrieval.invalidate_sources_cache()
    assert "kerberos" not in retrieval.corpus_vocabulary()[0]
