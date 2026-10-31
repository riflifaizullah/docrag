# Agent

`agent/docrag_agent.py` is a small tool-calling agent on top of DocRAG. You give it a question, and a local model decides which DocRAG tools to call, reads what they return, and writes the answer. It is useful for questions that span several documents or have several parts, which the plain question flow handles less well.

```bash
python agent/docrag_agent.py "What does the information security policy require for passwords, and how many days of annual leave do permanent employees get?"
```

```
step 1: search_passages({"question": "password requirements", "limit": 5})
step 2: search_passages({"question": "annual leave permanent employees days", "limit": 5})

Passwords must be at least 12 characters long ... (Information_Security_Policy_Sample.pdf, page 1) ...
Permanent employees are entitled to 12 days of annual leave ... (Employee_Handbook_Sample.pdf, page 1) ...
```

## How it works

1. The question and a short system prompt go to a local model through Ollama, together with the descriptions of three tools.
2. If the model asks for a tool, the agent runs it against the DocRAG backend and sends the result back. This repeats for up to 6 steps.
3. When the model answers instead of asking for a tool, that answer is printed. If the step limit is reached, the model is asked once more to answer from what it already has, without tools.

The tools are the same functions the [MCP server](MCP.md) uses: `list_documents`, `search_passages` and `ask_documents`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `DOCRAG_BACKEND_URL` | `http://127.0.0.1:8000` | The DocRAG backend |
| `DOCRAG_OLLAMA_HOST` | `http://127.0.0.1:11434` | Where Ollama is |
| `DOCRAG_OLLAMA_MODEL` | `qwen3.5:9b` | A model that supports tool calling |

It needs `requests` only, so it runs in the backend's Python environment (not the MCP one).

## Safety

- **Read-only.** The agent is given the three tools that read. `reindex_documents` is deliberately not offered, and a test checks that.
- **Bounded.** At most 6 tool-calling steps, and each tool result is cut at 6000 characters before the model sees it.
- **Errors are not fatal.** An unknown tool, bad arguments or an unreachable backend come back to the model as a message, so it can say what went wrong.
- Documents are untrusted text: a passage can contain instructions, and a model may follow them. The agent has no tool that changes anything or sends data anywhere, which limits the harm, but treat answers as you would any other model output.

## What has and has not been checked

- The loop has unit tests with a fake model: a direct answer, a tool call followed by an answer, an unknown tool, bad arguments, long results and the step limit.
- I ran it with `qwen3.5:9b` on two questions over the sample documents. It split a two-part question into two searches and answered both with sources, and for a question about a dress code and the GDPR it said the documents do not cover them instead of guessing.
- **It has not been benchmarked.** Two questions are not evidence of accuracy, and a small model can still pick the wrong tool or ignore a result. It is also slow on a CPU, because each step is a model call.
