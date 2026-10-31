# API reference

The backend listens on `127.0.0.1:8000` and has no sign-in, so keep it local. Requests with another `Host` header get 400. Interactive docs are available at http://localhost:8000/docs while it runs.

## Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How long must a password be?", "session_id": null}'
```

Request body:

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question, 1 to 4000 characters (422 otherwise). Several questions in one message are split and answered separately. The answer is written in the question's language |
| `session_id` | string or null | Omit or send null to start a new session, or pass an existing id to continue it |

Response shape:

```json
{
  "results": [
    {
      "question": "...",
      "answer": "...",
      "answered": true,
      "refused": false,
      "needs_clarification": false,
      "candidate_documents": [],
      "sources": ["Some_Document.pdf"],
      "log_id": 12
    }
  ],
  "answered": true,
  "refused": false,
  "needs_clarification": false,
  "candidate_documents": []
}
```

| Field | Meaning |
|---|---|
| `answered` | The system gave a real answer |
| `refused` | The documents do not contain the answer, so it declined |
| `needs_clarification` | The question matches several documents. `candidate_documents` lists them. Ask again naming one, for example `"<question> for document <file>"` (or `untuk dokumen` in Indonesian) |
| `sources` | Documents the answer was built from |
| `log_id` | Pass to `POST /flag/{log_id}` to flag the answer. Can be null |

## Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok", "model": "..."}` |
| `GET /ui` | `{"language": "en", "strings": {...}}`, the website's interface text |
| `GET /sources` | `{"sources": [...]}`, the names of indexed documents |
| `GET /documents` | Documents with page count, size and type label |
| `GET /documents/{file}/download` | The PDF itself |
| `GET /documents/{file}/preview` | Page count, a text snippet of the first page, and a thumbnail (only when Poppler is installed) |
| `GET /documents/{file}/chunks` | The chunks stored for a document, for debugging |
| `POST /search` | The best passages for a question, no language model. Body: `question` and optional `limit` (1 to 20, default 5). Returns `hits` (each with `source`, `page`, `text` cut to 800 characters, `score` and `rerank_score`) and `would_answer`, whether the relevance gate would let the question reach the language model. Used by the [MCP server](MCP.md) |
| `POST /sync` | Pick up new, changed and deleted files in the corpus folder. Returns `new`, `updated`, `unchanged` and `deleted` lists |
| `POST /sessions` | Create a chat session |
| `GET /sessions` | List sessions, newest first |
| `GET /sessions/{id}/history` | Messages of one session |
| `POST /flag/{log_id}` | Flag an answer. `category` is required and `corrected_answer` is optional, both sent as query parameters, not a JSON body |
| `GET /export_flags` | Export all flags |
| `GET /monitor`, `GET /monitor/status` | A small status page and its data |

The route definitions are in `backend/app/api.py`.
