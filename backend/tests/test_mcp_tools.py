"""The MCP server's tools call the backend and turn its problems into readable messages."""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp_server"))
import tools  # noqa: E402


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status
        self.ok = status < 400

    def json(self):
        return self._payload


@pytest.fixture
def backend(monkeypatch):
    """Records the calls and answers with whatever the test sets in `backend.reply`."""
    calls = []

    class Backend:
        reply = FakeResponse({})

    def fake_request(method, url, timeout, **kwargs):
        calls.append((method, url, timeout, kwargs))
        if isinstance(Backend.reply, Exception):
            raise Backend.reply
        return Backend.reply

    monkeypatch.setattr(tools.requests, "request", fake_request)
    Backend.calls = calls
    return Backend


def test_list_documents_returns_the_documents(backend):
    backend.reply = FakeResponse({"documents": [{"filename": "a.pdf"}]})
    assert tools.list_documents() == [{"filename": "a.pdf"}]
    assert backend.calls[0][:2] == ("GET", f"{tools.BACKEND_URL}/documents")


def test_search_passages_sends_the_question_and_keeps_the_limit_in_range(backend):
    backend.reply = FakeResponse({"hits": [], "would_answer": False})
    tools.search_passages("question", limit=500)
    assert backend.calls[0][3]["json"] == {"question": "question", "limit": 20}
    tools.search_passages("question", limit=0)
    assert backend.calls[1][3]["json"]["limit"] == 1


def test_ask_documents_keeps_only_what_an_assistant_needs(backend):
    backend.reply = FakeResponse({"results": [{
        "question": "q", "answer": "a", "sources": ["x.pdf"], "refused": False, "needs_clarification": False,
        "candidate_documents": [], "answered": True, "log_id": 7,
    }]})
    answers = tools.ask_documents("q")["answers"]
    assert answers == [{"question": "q", "answer": "a", "sources": ["x.pdf"], "refused": False,
                        "needs_clarification": False, "candidate_documents": []}]
    assert backend.calls[0][2] == tools.ASK_TIMEOUT_SECONDS  # a local model is slow


def test_reindex_posts_to_sync(backend):
    backend.reply = FakeResponse({"new": []})
    assert tools.reindex_documents() == {"new": []}
    assert backend.calls[0][:2] == ("POST", f"{tools.BACKEND_URL}/sync")


def test_unreachable_backend_says_how_to_start_it(backend):
    backend.reply = requests.exceptions.ConnectionError()
    with pytest.raises(tools.BackendError, match="not reachable"):
        tools.list_documents()


def test_timeout_is_reported(backend):
    backend.reply = requests.exceptions.Timeout()
    with pytest.raises(tools.BackendError, match="did not answer"):
        tools.search_passages("q")


def test_rejected_request_is_explained(backend):
    backend.reply = FakeResponse(status=422)
    with pytest.raises(tools.BackendError, match="rejected"):
        tools.search_passages("")


def test_other_http_errors_do_not_leak_details(backend):
    backend.reply = FakeResponse(status=500)
    with pytest.raises(tools.BackendError, match="HTTP 500"):
        tools.ask_documents("q")
