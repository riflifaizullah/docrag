# Public benchmark

A small, reproducible test of DocRAG's retrieval, run on public documents. Anyone can repeat it.

```bash
pip install -r backend/requirements.txt
python benchmark/fetch_documents.py       # downloads 9 PDFs and checks their SHA-256
python benchmark/run_benchmark.py         # indexes them three times, asks 99 questions in 6 variants
```

A run takes well under an hour on a laptop CPU, mostly indexing. It calls no language model, so the numbers are the same every time. The vision model for diagram pages is switched off (`DOCRAG_VLM=0`).

## What is in it

- **Documents (9):** seven NIST publications in English (about 530 pages) and two Indonesian laws (about 150 pages). NIST publications are US government works, and Indonesian laws are not protected by copyright. The PDFs are not stored in this repository, `documents.json` lists the sources and checksums.
- **Answerable questions (79):**
  - 64 *heading* questions, made automatically: "What does the document say about *a section title*?". The right page is the page where the section starts (or the next one). These share words with the text, so they are the easy kind.
  - 15 *hand-written* questions in plain wording ("What is the minimum length of a password that a user picks themselves?"). The right pages are every page that contains the answer text, so the ground truth can be audited in `questions_manual.json`.
- **Questions that should be refused (20):** 14 off-topic ones (cooking, sports, general knowledge) and 6 *near misses* that sound related but are not covered (for example a GDPR fine, when the documents are NIST guides and an Indonesian data protection law).

## Results

| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |
|---|---|---|---|---|---|---|
| full (default) | 100% (79/79) | 99% (78/79) | 84% (66/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |
| no keyword search | 99% (78/79) | 95% (75/79) | 82% (65/79) | 95% (78/82) | 99% (78/79) | 80% (16/20) |
| no reranker | 99% (78/79) | 91% (72/79) | 58% (46/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| vector search only | 99% (78/79) | 73% (58/79) | 53% (42/79) | 96% (74/77) | 94% (74/79) | 85% (17/20) |
| chunks of 500 characters | 100% (79/79) | 99% (78/79) | 82% (65/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |
| chunks of 2000 characters | 100% (79/79) | 99% (78/79) | 89% (70/79) | 95% (79/83) | 100% (79/79) | 80% (16/20) |

By question type (default variant): heading questions 64/64 for both the document and the page, hand-written questions 15/15 for the document and 14/15 for the page.

## What the numbers say

- **The reranker and keyword search matter at page level.** Vector search alone puts the right page in the top 8 for 73 percent of questions. Adding keyword search gives 91 percent, and adding the reranker as well gives 99 percent. The reranker matters most for getting the right page first: 58 percent without it against 84 percent with it (both with keyword search).
- **Chunk size made little difference.** The 500, 1000 and 2000 character variants differ by a few questions, which is within noise on 79 questions. Chunks of 2000 characters put the right page first more often (89 percent against 82 and 84), but I would not draw a conclusion from this sample.
- **The relevance gate lets near misses through.** All 14 off-topic questions were refused. Four of the 6 near-miss questions were not (the HIPAA, GDPR and TLS ones, and the Indonesian GDPR one). The gate cannot tell "related topic" from "covered by these documents", so refusing those is left to the language model and its prompt. This is the main weakness in this benchmark.

## What this does not show

- **Document-level retrieval is not tested hard.** With only 9 documents, finding the right document is easy, and nearly every variant gets 99 to 100 percent. A larger collection would separate the variants more.
- **No answers are graded.** The test stops at retrieval and the relevance gate. It does not check what the language model writes. "Gate precision" is the share of questions that passed the gate that should have passed, and "gate recall" is the share of answerable questions that passed it.
- **The sample is small.** There are 79 answerable and 20 refusal questions. A difference of one to four questions is noise, and no confidence intervals are given.
- **The heading questions are easy by construction**, and they are 64 of the 79. The hand-written ones are a more honest test but there are only 15.
- **Only the retrieval step is measured**, not the rest of the question handling (named documents, ambiguity checks, listings).
- **Not comparable with the README's earlier figures.** Those came from a private collection, a different question set and a different measure.

## Files

| File | Purpose |
|---|---|
| `documents.json` | Sources, licences and SHA-256 of the documents |
| `fetch_documents.py` | Downloads and verifies them |
| `build_questions.py` | Generates `questions_headings.json` from the indexed headings |
| `questions_manual.json` | Hand-written questions and the questions that should be refused |
| `run_benchmark.py` | Indexes, asks every question in each variant, writes `results.md` and `results.json` |
