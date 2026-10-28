# API reference

The backend listens on port 8000. Interactive docs are available at http://localhost:8000/docs while it runs.

## Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Berapa minimal panjang kata sandi?", "session_id": null}'
```

Request body:

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question. Several questions in one message are split and answered separately |
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
| `needs_clarification` | The question matches several documents. `candidate_documents` lists them. Ask again naming one, for example `"<question> untuk dokumen <file>"` |
| `sources` | Documents the answer was built from |
| `log_id` | Pass to `POST /flag/{log_id}` to flag the answer. Can be null |

## Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok", "model": "..."}` |
| `GET /sources` | `{"sources": [...]}`, the names of indexed documents |
| `GET /documents` | Documents with page count, size and type label |
| `GET /documents/{file}/download` | The PDF itself |
| `GET /documents/{file}/preview` | Page count, a text snippet of the first page, and a thumbnail (only when Poppler is installed) |
| `GET /documents/{file}/chunks` | The chunks stored for a document, for debugging |
| `POST /sync` | Pick up new, changed and deleted files in `backend/data/corpus/` |
| `POST /sessions` | Create a chat session |
| `GET /sessions` | List sessions, newest first |
| `GET /sessions/{id}/history` | Messages of one session |
| `POST /flag/{log_id}` | Flag an answer. `category` is required and `corrected_answer` is optional, both sent as query parameters, not a JSON body |
| `GET /export_flags` | Export all flags |
| `GET /monitor`, `GET /monitor/status` | A small status page and its data |

The route definitions are in `backend/app/api.py`.
