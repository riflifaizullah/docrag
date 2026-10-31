"""A small tool-calling agent over DocRAG.

    python agent/docrag_agent.py "Which document covers passwords, and how does it differ from the handbook?"

A local model (through Ollama) decides which DocRAG tools to call, reads what they return, and answers. It can
search passages, list documents and ask for a written answer, in as many steps as it needs up to a limit. It cannot
change anything: re-indexing is not offered to it. The backend must be running (DOCRAG_BACKEND_URL).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_server"))
import tools  # noqa: E402

OLLAMA_HOST = os.getenv("DOCRAG_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("DOCRAG_OLLAMA_MODEL", "qwen3.5:9b")
MAX_STEPS = 6
MAX_TOOL_RESULT_CHARS = 6000

SYSTEM_PROMPT = (
    "You answer questions about the user's own PDF documents using tools. Use list_documents to see what exists, "
    "search_passages to read the exact passages and pages, and ask_documents for a written answer from one question. "
    "Break a question that spans several documents into separate searches. Base every statement on tool results and "
    "name the document and page. If the tools show the documents do not cover the question, say so plainly and do "
    "not guess."
)

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_documents", "description": "List the indexed documents with type and page count.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_passages",
        "description": "Find the passages that best match a short question, with document and page. Fast.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "A focused question or key phrase."},
            "limit": {"type": "integer", "description": "How many passages, 1 to 20. Default 5."}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "ask_documents",
        "description": "Get a written answer, with sources, to one question about the documents. Slow.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]

# The agent only gets tools that read. reindex_documents is deliberately left out.
TOOL_FUNCTIONS = {
    "list_documents": lambda **_: {"documents": tools.list_documents()},
    "search_passages": lambda question, limit=5, **_: tools.search_passages(question, int(limit)),
    "ask_documents": lambda question, **_: tools.ask_documents(question),
}


def ollama_chat(messages: list[dict], with_tools: bool = True) -> dict:
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False, "options": {"temperature": 0}}
    if with_tools:
        body["tools"] = TOOL_SPECS
    response = requests.post(f"{OLLAMA_HOST}/api/chat", json=body, timeout=600)
    response.raise_for_status()
    return response.json()["message"]


def run_tool(name: str, arguments: dict) -> str:
    """The result of one tool call as text for the model. Problems become messages it can react to."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return json.dumps({"error": f"There is no tool called {name}. Use one of: {', '.join(TOOL_FUNCTIONS)}."})
    try:
        result = function(**arguments)
    except tools.BackendError as exc:
        return json.dumps({"error": str(exc)})
    except TypeError as exc:
        return json.dumps({"error": f"The arguments for {name} were not valid: {exc}"})
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= MAX_TOOL_RESULT_CHARS else text[:MAX_TOOL_RESULT_CHARS] + " ...[cut]"


def run_agent(question: str, chat: Callable[..., dict] = ollama_chat, max_steps: int = MAX_STEPS) -> dict:
    """Returns {"answer": str, "steps": [{"tool", "arguments", "result"}], "finished": bool}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps = []
    for _ in range(max_steps):
        message = chat(messages)
        calls = message.get("tool_calls") or []
        if not calls:
            return {"answer": (message.get("content") or "").strip(), "steps": steps, "finished": True}
        messages.append(message)
        for call in calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or {}
            result = run_tool(name, arguments)
            steps.append({"tool": name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": result})
    # Out of steps: ask once more without letting it call tools, so the person still gets an answer.
    messages.append({"role": "user", "content": "Answer now from what the tools returned. Do not call more tools."})
    final = chat(messages, False)
    return {"answer": (final.get("content") or "").strip(), "steps": steps, "finished": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    args = parser.parse_args()
    outcome = run_agent(args.question)
    for number, step in enumerate(outcome["steps"], 1):
        print(f"step {number}: {step['tool']}({json.dumps(step['arguments'], ensure_ascii=False)})")
    if not outcome["finished"]:
        print(f"(stopped after {MAX_STEPS} steps)")
    print("\n" + outcome["answer"])


if __name__ == "__main__":
    main()
