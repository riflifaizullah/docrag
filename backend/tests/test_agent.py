"""The agent loop, driven by a fake model so no Ollama is needed."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
import docrag_agent as agent  # noqa: E402


def call(name, **arguments):
    return {"function": {"name": name, "arguments": arguments}}


class FakeModel:
    """Plays back the messages it was given, one per request, and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, messages, with_tools=True):
        self.requests.append((list(messages), with_tools))
        return self.replies.pop(0)


@pytest.fixture
def backend(monkeypatch):
    calls = []
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_passages", lambda question, limit=5, **_: calls.append(question) or {"hits": [{"source": "a.pdf", "page": 1}]})
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "list_documents", lambda **_: {"documents": []})
    return calls


def test_answers_directly_when_no_tool_is_needed():
    model = FakeModel({"role": "assistant", "content": "  Hello.  "})
    outcome = agent.run_agent("hi", chat=model)
    assert outcome == {"answer": "Hello.", "steps": [], "finished": True}


def test_calls_a_tool_reads_the_result_then_answers(backend):
    model = FakeModel(
        {"role": "assistant", "content": "", "tool_calls": [call("search_passages", question="password length")]},
        {"role": "assistant", "content": "At least 12 characters (a.pdf, p. 1)."},
    )
    outcome = agent.run_agent("How long must a password be?", chat=model)
    assert backend == ["password length"]
    assert outcome["finished"] and "12 characters" in outcome["answer"]
    assert outcome["steps"][0]["tool"] == "search_passages"
    # the second request carries the tool result back to the model
    second = model.requests[1][0]
    assert second[-1]["role"] == "tool" and "a.pdf" in second[-1]["content"]


def test_an_unknown_tool_becomes_a_message_the_model_can_react_to():
    model = FakeModel(
        {"role": "assistant", "tool_calls": [call("delete_everything")]},
        {"role": "assistant", "content": "I cannot do that."},
    )
    outcome = agent.run_agent("x", chat=model)
    assert "no tool called delete_everything" in outcome["steps"][0]["result"]
    assert outcome["answer"] == "I cannot do that."


def test_reindex_is_not_available_to_the_agent():
    assert "reindex_documents" not in agent.TOOL_FUNCTIONS
    assert "reindex_documents" not in [spec["function"]["name"] for spec in agent.TOOL_SPECS]


def test_wrong_arguments_and_backend_errors_are_reported_not_raised(monkeypatch):
    def unreachable(**_):
        raise agent.tools.BackendError("not reachable")

    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "list_documents", unreachable)
    assert json.loads(agent.run_tool("list_documents", {}))["error"] == "not reachable"
    assert "not valid" in json.loads(agent.run_tool("search_passages", {"wrong": 1}))["error"]


def test_long_tool_results_are_cut(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "list_documents", lambda **_: {"documents": ["x" * 20000]})
    result = agent.run_tool("list_documents", {})
    assert len(result) < agent.MAX_TOOL_RESULT_CHARS + 20 and result.endswith("[cut]")


def test_stops_after_the_step_limit_and_still_gives_an_answer(backend):
    looping = {"role": "assistant", "tool_calls": [call("search_passages", question="again")]}
    model = FakeModel(looping, looping, {"role": "assistant", "content": "Best I found: a.pdf p. 1."})
    outcome = agent.run_agent("x", chat=model, max_steps=2)
    assert outcome["finished"] is False
    assert len(outcome["steps"]) == 2
    assert outcome["answer"] == "Best I found: a.pdf p. 1."
    messages, with_tools = model.requests[-1]
    assert with_tools is False and "Do not call more tools" in messages[-1]["content"]
