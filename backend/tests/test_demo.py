"""Demo mode: read-only, no language model, rate limited."""
import pytest
from fastapi import HTTPException

from app import config, demo, generation, retrieval
from app.lang import en as en_lang
from app.lang import id as id_lang


def hit(text, source="Policy.pdf", page=1, rerank=5.0):
    return {"text": text, "source": source, "page": page, "score": 0.8, "rerank_score": rerank}


@pytest.fixture
def demo_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)


def test_rate_limiter_allows_up_to_the_limit_then_waits():
    now = [0.0]
    limiter = demo.RateLimiter(2, 60, clock=lambda: now[0])
    assert limiter.allow() and limiter.allow()
    assert not limiter.allow()
    now[0] = 61
    assert limiter.allow()  # the old calls fell out of the window


def test_changing_endpoints_are_blocked_in_demo_mode_only(monkeypatch):
    demo.require_writable()  # normal mode: fine
    monkeypatch.setattr(config, "DEMO_MODE", True)
    with pytest.raises(HTTPException) as blocked:
        demo.require_writable()
    assert blocked.value.status_code == 403


def test_questions_are_rate_limited_in_demo_mode(demo_mode, monkeypatch):
    monkeypatch.setattr(demo, "limiter", demo.RateLimiter(1, 60))
    demo.check_rate_limit()
    with pytest.raises(HTTPException) as busy:
        demo.check_rate_limit()
    assert busy.value.status_code == 429


def test_rate_limit_is_off_outside_demo_mode(monkeypatch):
    monkeypatch.setattr(demo, "limiter", demo.RateLimiter(0, 60))
    demo.check_rate_limit()  # would be refused if it applied


def test_answer_is_the_best_passages_with_their_pages(monkeypatch):
    hits = [hit("Passwords must be at least 12 characters long.", page=1), hit("Change them every 90 days.", page=2)]
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: True)
    answer = demo.passages_answer("How long must a password be?", hits)
    assert answer["answered"] and not answer["refused"]
    assert answer["sources"] == ["Policy.pdf"]
    assert "at least 12 characters" in answer["answer"] and "(Policy.pdf, p. 2)" in answer["answer"]
    assert answer["answer"].startswith(en_lang.DEMO_ANSWER_INTRO)


def test_weak_passages_are_not_listed_next_to_a_strong_one(monkeypatch):
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: True)
    hits = [hit("Strong match.", rerank=4.0), hit("Near match.", source="B.pdf", rerank=2.0), hit("Weak match.", source="C.pdf", rerank=-5.0)]
    answer = demo.passages_answer("a question", hits)
    assert "Strong match" in answer["answer"] and "Near match" in answer["answer"] and "Weak match" not in answer["answer"]
    assert answer["sources"] == ["Policy.pdf", "B.pdf"]


def test_a_question_the_gate_rejects_gets_the_usual_refusal(monkeypatch):
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: False)
    answer = demo.passages_answer("How does Kerberos work?", [hit("unrelated", rerank=-4)])
    assert answer["refused"] and not answer["answered"] and answer["sources"] == []
    assert answer["answer"] == en_lang.NOT_FOUND


def test_long_passages_are_shortened_without_cutting_a_word(monkeypatch):
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: True)
    answer = demo.passages_answer("question about words", [hit("word " * 400)])
    line = answer["answer"].splitlines()[1]
    assert "..." in line and len(line) < demo.PASSAGE_CHARS + 60


def test_the_answer_follows_the_language_of_the_question(monkeypatch):
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: True)
    answer = demo.passages_answer("Berapa hari cuti tahunan yang didapat karyawan tetap?", [hit("12 hari cuti.")])
    assert answer["answer"].startswith(id_lang.DEMO_ANSWER_INTRO)


def test_retrieval_is_used_when_no_hits_are_given(monkeypatch):
    seen = {}
    monkeypatch.setattr(retrieval, "retrieve", lambda q, k=8, **_: seen.update(q=q, k=k) or [hit("text")])
    monkeypatch.setattr(generation, "passes_relevance_gate", lambda h, q: True)
    demo.passages_answer("a question")
    assert seen == {"q": "a question", "k": demo.PASSAGES_SHOWN}
