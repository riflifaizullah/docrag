# Getting started

This guide takes you from a clean machine to asking your first question, using the bundled sample documents.

## What you need

| Requirement | Notes |
|---|---|
| Python 3.10 or newer | The backend |
| .NET 8 SDK | The website |
| [Ollama](https://ollama.com) | Runs the language model locally |
| A GPU with about 8 GB of memory | Recommended. CPU works but is slow |
| Tesseract and Poppler | Optional. Only needed for scanned PDFs |

## 1. Install

```bash
pip install -r backend/requirements.txt
ollama pull qwen3.5:9b
```

The first run also downloads the embedding model and the reranker (about 600 MB in total) from Hugging Face.

## 2. Load the sample documents

```bash
python scripts/load_samples.py
```

This copies the five English sample PDFs from `samples/en/` into `backend/data/corpus/` and builds the index. The samples are fictional policy documents for an invented company: an employee handbook, and policies for leave, business travel, information security and procurement. For the Indonesian set run `python scripts/load_samples.py id`. Both sets can live in one index. You can rebuild the PDFs with `python scripts/generate_sample_docs.py [en|id|all]` (needs `pip install -r requirements-dev.txt`).

To use your own documents instead, put PDFs in `backend/data/corpus/` and click the re-index button in the sidebar, or run `cd backend && python -m app.sync_documents`. The button reports how many files were new, updated or removed. The corpus folder is checked when the backend starts and then every 15 minutes.

## 3. Start the app

```bash
cd dotnet/RagSystemWeb
dotnet run
```

Open http://localhost:5080. The website starts the backend for you. If it cannot find your Python, set the `DOCRAG_PYTHON` environment variable to the full path of the interpreter that has the requirements installed.

To run the backend alone: `cd backend && uvicorn app.api:app --host 127.0.0.1 --port 8000`, then open http://localhost:8000/monitor.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `DOCRAG_PYTHON` | `python` | Interpreter the website uses to start the backend |
| `DOCRAG_LANG` | `en` | Language used when a question's language is unclear |
| `DOCRAG_DATA_DIR` | `backend/data` | Where documents, the index, sessions and logs live. Use different folders to keep several indexes |
| `DOCRAG_API_HOST` | `127.0.0.1` | Address the backend listens on. It has no sign-in, so keep it local |
| `DOCRAG_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Host names the backend accepts |
| `DOCRAG_OLLAMA_HOST` | `http://127.0.0.1:11434` | Where the backend finds Ollama |
| `DOCRAG_OLLAMA_MODEL` | `qwen3.5:9b` | Language model the backend asks Ollama for |
| `DOCRAG_DEMO` | `0` | Set to `1` for a read-only demo with no language model (see [DEPLOY.md](DEPLOY.md)) |
| `DOCRAG_VLM` | `1` | Set to `0` to skip the vision model that describes diagram pages |
| `DOCRAG_CHUNK_SIZE` | `1000` | Characters per passage when indexing (the overlap is 15 percent of it). Re-index after changing it |
| `DOCRAG_BACKEND_URL` | not set | Website only. If set, the website uses that backend and never starts one |

## Run with Docker

`docker-compose.yml` starts four containers: Ollama, a one-time step that pulls the language model, the backend and the website.

```bash
docker compose up --build
```

Then open http://localhost:5080. The first start downloads the language model (about 6 GB) and the embedding and reranker models (about 600 MB), then indexes the English samples, so give it several minutes. Later starts reuse the downloads, which live in Docker volumes.

- **Your own documents:** `DOCRAG_DOCS=/path/to/pdfs docker compose up --build`. The folder is mounted read-only. Add or change a PDF and click the re-index button, or wait for the 15-minute check.
- **Indonesian default:** `DOCRAG_LANG=id docker compose up --build`.
- **Ports:** 5080 (website) and 8000 (backend, which the PDF preview uses) are published on `127.0.0.1` only, because the backend has no sign-in. Do not change that to `0.0.0.0` on a shared machine.
- **Speed:** Ollama runs on the CPU here, so answers can take a minute or more. With an NVIDIA GPU, add a `deploy.resources.reservations.devices` entry to the `ollama` service, see the Docker documentation on GPU support.
- **The vision model** that describes diagram pages is not pulled. Pages that are mostly diagrams are indexed without a description. Add `qwen2.5vl:7b` to the `models` step if you need it.
- **Reset everything:** `docker compose down -v` removes the index and the downloaded models.

The Docker setup was written for Docker Compose v2 and is built (but not run end to end) in CI.

## 4. Things to try

The interface is in English. Answers follow the language of your question, and the English samples are loaded by default.

| Ask | What to expect |
|---|---|
| `How many days of annual leave do permanent employees get?` | An answer (12 days) with its source. The employee handbook and the leave policy both cover this, so the other one is offered as an alternative |
| `What is the daily allowance for domestic business travel?` | An answer from the travel policy |
| `What does Appendix 1 say?` | A list of all five documents to choose from, because every sample has an "Appendix 1" |
| `Is there a dress code?` | A refusal: the documents do not cover it |
| `List all appendices in Leave_Policy_Sample.pdf` | A deterministic list of its appendices |
| `Summarize Business_Travel_Policy_Sample.pdf` | A summary built from the whole document |

With the Indonesian samples loaded (or both sets), ask in Indonesian and the answer comes back in Indonesian:

| Ask | What to expect |
|---|---|
| `Berapa hari cuti tahunan yang didapat karyawan tetap?` | An answer from the leave policy, with its source |
| `Berapa lama batas waktu mengajukan klaim biaya perjalanan dinas?` | An answer from the travel policy |
| `Berapa minimal panjang kata sandi?` | An answer from the information security policy |
| `Apa isi Lampiran 1?` | A list of all five documents to choose from, because every sample has a "Lampiran 1" |
| `Apa penyebab terjadinya gerhana matahari total?` | A refusal: the documents do not cover it |
| `Apa isi Pasal 2 pada dokumen Kebijakan_Keamanan_Informasi_Contoh.pdf?` | An answer scoped to the document you named |

The wording of a generated answer varies from run to run, because it comes from the language model. The retrieval behavior, meaning which document is found, the document choice list and the refusal, does not depend on the model's wording.

Click a source chip under an answer to open the PDF in the side panel. Use the flag button to mark an answer as wrong.

## Settings and troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| The page loads but answers fail | Ollama is not running, or the model is not pulled. Run `ollama pull qwen3.5:9b` and check `ollama list` |
| The first question is very slow | The models load on first use. Later questions are faster |
| A reasonable question is refused | The relevance gate is strict, especially on a small document set, and retrieval depends on vocabulary ("hotel" versus "accommodation"). Try the document's own wording, or lower `RELEVANCE_MIN_SCORE` and `RELEVANCE_MIN_RERANK_SCORE` in `backend/app/config.py` |
| "0 documents" in the sidebar | No PDFs are indexed yet. Run the load command above, or add PDFs and click the re-index button |
| "Could not load" in the sidebar | The backend is not running. Start it, then click Retry |
| The website cannot start the backend | Set `DOCRAG_PYTHON`, or start the backend yourself on port 8000 |
| Scanned PDFs produce no text | Install Tesseract and Poppler so OCR can run |
| Port 5080 or 8000 is in use | Stop the other program, or change the port in `dotnet/RagSystemWeb/Properties/launchSettings.json` |

## Running the tests

```bash
cd backend && python -m pytest
cd dotnet && dotnet test RagSystemWeb.Tests
```

The backend tests do not need Ollama. The website tests use a fake backend client, so no server is needed either.
