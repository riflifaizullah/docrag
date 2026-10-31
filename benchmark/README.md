# Public benchmark

A small, reproducible test of DocRAG's retrieval, run on public documents. Anyone can repeat it.

```bash
pip install -r backend/requirements.txt
python benchmark/fetch_documents.py       # downloads 9 PDFs and checks their SHA-256
python benchmark/run_benchmark.py         # indexes them three times, asks 99 questions in 6 variants
```

A run takes well under an hour on a laptop CPU, mostly indexing. It calls no language model, so nothing in the question step is random. The vision model for diagram pages is switched off (`DOCRAG_VLM=0`).

## What is in it

- **Documents (9):** seven NIST publications in English (about 530 pages) and two Indonesian laws (about 150 pages). NIST publications are US government works, and Indonesian laws are not protected by copyright. The PDFs are not stored in this repository, `documents.json` lists the sources and checksums.
- **Answerable questions (79):**
  - 64 *heading* questions, made automatically: "What does the document say about *a section title*?". The right page is the page where the section starts (or the next one). These share words with the text, so they are the easy kind.
  - 15 *hand-written* questions in plain wording ("What is the minimum length of a password that a user picks themselves?"). The right pages are every page that contains the answer text, so the ground truth can be audited in `questions_manual.json`.
- **Questions that should be refused (20):** 14 off-topic ones (cooking, sports, general knowledge) and 6 *near misses* that sound related but are not covered (for example a GDPR fine, when the documents are NIST guides and an Indonesian data protection law).

## Results

| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |
|---|---|---|---|---|---|---|
| full (default) | 100% (79/79) | 99% (78/79) | 84% (66/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| no keyword search | 99% (78/79) | 95% (75/79) | 82% (65/79) | 99% (78/79) | 99% (78/79) | 95% (19/20) |
| no reranker | 99% (78/79) | 91% (72/79) | 59% (47/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| vector search only | 96% (76/79) | 76% (60/79) | 54% (43/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| chunks of 500 characters | 100% (79/79) | 99% (78/79) | 82% (65/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| chunks of 2000 characters | 100% (79/79) | 99% (78/79) | 89% (70/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |

By question type (default variant): heading questions 64/64 for both the document and the page, hand-written questions 15/15 for the document and 14/15 for the page.

## What the numbers say

- **The reranker and keyword search matter at page level.** Vector search alone puts the right page in the top 8 for about 76 percent of questions. Adding keyword search gives 91 percent, and adding the reranker as well gives 99 percent. The reranker matters most for getting the right page first: 59 percent without it against 84 percent with it (both with keyword search).
- **Chunk size made little difference.** The 500, 1000 and 2000 character variants differ by a few questions, which is within noise on 79 questions. Chunks of 2000 characters put the right page first more often (89 percent against 82 and 84), but I would not draw a conclusion from this sample.
- **Off-topic and near-miss questions are now mostly refused**, after a change to the relevance gate described below. 19 of 20 were refused. The one that still gets through asks which encryption algorithm NIST recommends for TLS 1.3, and the documents do mention TLS, so it is arguably a fair near miss.

## The relevance gate change

The first run of this benchmark found that the relevance gate let 4 of the 6 near-miss questions through (a GDPR fine, HIPAA, TLS 1.3 ciphers, a GDPR question in Indonesian), because the nearest passages looked similar enough.

The fix: if the question contains a specific word that appears nowhere in the indexed documents (for example "GDPR" or "Kerberos"), the question passes only when the best reranker score is at least 3. A very strong match still overrides the check. A word counts as specific if it is an acronym or if the embedding model has to cut it into three or more pieces. Everyday words such as "minimum" or "length" never trigger the check, because a document may word them differently. Simple word endings are ignored, so "memorizing" matches "memorized" and the Indonesian "datanya" matches "data". The word list is read from whatever is indexed, so it follows the documents when files are added or removed (this has a test, and I also checked it by adding and removing a PDF about Kerberos). The check needs the reranker, so it does nothing when the reranker is switched off.

To avoid tuning the rule on the questions it is reported on, I wrote a second set of 95 questions (`questions_dev.json`, plus `questions_dev_headings.json`: 50 headings, 15 hand-written questions, 30 that should be refused) and chose the settings on that set only. The 99 test questions were not used for tuning.

| | Gate precision | Gate recall | Near misses refused |
|---|---|---|---|
| Tuning set, before | 94% (64/68) | 98% (64/65) | 21/25 |
| Tuning set, after | 97% (64/66) | 98% (64/65) | 23/25 |
| Test set, before | 95% (79/83) | 100% (79/79) | 2/6 |
| Test set, after | 99% (79/80) | 100% (79/79) | 5/6 |

No answerable test question was lost. (My first version of the rule was stricter and did lose one, a plain-wording question about the minimum password length, and the same mistake showed up when I tried the sample documents. Restricting the check to specific words fixed both.) The one answerable tuning question that is lost was already lost before the change.

The risk of the check remains: a question that uses a rare word the documents never use, such as a synonym or a misspelling, can be refused although the answer is there, unless the best match is very strong. The thresholds were chosen on 95 questions from these 9 documents, so they may need adjusting for very different collections.

## What this does not show

- **Document-level retrieval is not tested hard.** With only 9 documents, finding the right document is easy, and nearly every variant gets 96 to 100 percent. A larger collection would separate the variants more.
- **No answers are graded.** The test stops at retrieval and the relevance gate. It does not check what the language model writes. "Gate precision" is the share of questions that passed the gate that should have passed, and "gate recall" is the share of answerable questions that passed it.
- **The sample is small.** There are 79 answerable and 20 refusal questions. A difference of one to four questions is noise, and no confidence intervals are given.
- **Indexing is not exactly repeatable.** The vector index uses approximate search, so two builds of the same documents can differ slightly. Across three builds I saw the vector-only variant move between 73 and 77 percent for the right page in the top 8. Treat differences of a few points as noise.
- **The heading questions are easy by construction**, and they are 64 of the 79. The hand-written ones are a more honest test but there are only 15.
- **Only the retrieval step is measured**, not the rest of the question handling (named documents, ambiguity checks, listings).
- **Not comparable with the README's earlier figures.** Those came from a private collection, a different question set and a different measure.

## Files

| File | Purpose |
|---|---|
| `documents.json` | Sources, licences and SHA-256 of the documents |
| `fetch_documents.py` | Downloads and verifies them |
| `build_questions.py` | Generates `questions_headings.json` (or `questions_dev_headings.json` with `dev`) from the indexed headings |
| `questions_manual.json` | Hand-written test questions and the test questions that should be refused |
| `questions_dev.json`, `questions_dev_headings.json` | The separate set used to choose the gate setting |
| `run_benchmark.py` | Indexes, asks every test question in each variant, writes `results.md` and `results.json` |
