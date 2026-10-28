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

This copies the five sample PDFs from `samples/` into `backend/data/corpus/` and builds the index. The samples are fictional policy documents for an invented company: an employee handbook, and policies for leave, business travel, information security and procurement. You can rebuild them with `python scripts/generate_sample_docs.py` (needs `pip install -r requirements-dev.txt`).

To use your own documents instead, put PDFs in `backend/data/corpus/` and run `cd backend && python -m app.sync_documents`.

## 3. Start the app

```bash
cd dotnet/RagSystemWeb
dotnet run
```

Open http://localhost:5080. The website starts the backend for you. If it cannot find your Python, set the `DOCRAG_PYTHON` environment variable to the full path of the interpreter that has the requirements installed.

To run the backend alone: `cd backend && uvicorn app.api:app --port 8000`, then open http://localhost:8000/monitor.

## 4. Things to try

The interface and the documents are in Indonesian, so ask in Indonesian.

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
| A reasonable question is refused | The relevance gate is strict, especially on a small document set. Try more specific wording, or lower `RELEVANCE_MIN_SCORE` and `RELEVANCE_MIN_RERANK_SCORE` in `backend/app/config.py` |
| "0 documents" in the sidebar | No PDFs are indexed yet. Run the load or sync command above |
| The website cannot start the backend | Set `DOCRAG_PYTHON`, or start the backend yourself on port 8000 |
| Scanned PDFs produce no text | Install Tesseract and Poppler so OCR can run |
| Port 5080 or 8000 is in use | Stop the other program, or change the port in `dotnet/RagSystemWeb/Properties/launchSettings.json` |

## Running the tests

```bash
cd backend && python -m pytest
cd dotnet && dotnet test RagSystemWeb.Tests
```

The backend tests do not need Ollama. The website tests use a fake backend client, so no server is needed either.
