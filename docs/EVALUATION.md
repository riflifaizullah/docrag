# Evaluation

How well does DocRAG find the right passage, and how well does it refuse questions the documents cannot answer? This page gathers every measurement in one place, says what each number means, and says what it does not show.

## In short

- **Finding the right page works well on the public test.** On 9 public documents, the right page was among the 8 passages sent to the language model for 99 percent of the 79 answerable questions (78 of 79). Vector search alone managed about 76 percent. Keyword search and the reranker account for the difference.
- **Refusing is good but not perfect.** 19 of 20 questions that should be refused were refused, 14 of 14 off-topic ones and 5 of 6 near misses (a related topic the documents do not cover). A first version of the gate refused only 2 of those 6, and one change fixed most of that.
- **The test is small and measures retrieval only.** It does not grade the answers the language model writes. Treat it as evidence that the retrieval design works, not as an accuracy score for the whole product.
- **An earlier private run** on more than 1,000 real documents showed the same shape at scale. It cannot be shared or reproduced, so it carries less weight.

## 1. The public benchmark (reproducible)

Everything needed to repeat it is in [`benchmark/`](../benchmark/README.md).

```bash
python benchmark/fetch_documents.py
python benchmark/run_benchmark.py
```

**Documents (9).** Seven NIST publications in English (about 530 pages) and two Indonesian laws (about 150 pages). The PDFs are downloaded, not stored in the repository, and each file is checked against a recorded SHA-256.

**Questions (99).**

| Kind | Count | Notes |
|---|---|---|
| Heading questions | 64 | Generated from section titles ("What does the document say about ..."). The right page is where the section starts. Easy, because they share words with the text. |
| Hand-written questions | 15 | Plain wording, such as "What is the minimum length of a password that a user picks themselves?". Every page containing the answer text counts as right. |
| Off-topic, should be refused | 14 | Cooking, sports, general knowledge, in English and Indonesian. |
| Near misses, should be refused | 6 | Related topics the documents do not cover, for example a GDPR fine. |

**What is measured.** For each question the real retrieval code runs (vector search, keyword search, reranker, relevance gate) and keeps the top 8 passages. No language model is called, so a rerun gives the same question results.

- **Right document in top 8:** at least one of the 8 passages comes from the correct document.
- **Right page in top 8:** at least one comes from a correct page of it.
- **Right page first:** the top passage is from a correct page.
- **Gate precision:** of the questions that passed the relevance gate, the share that should have passed. A question that should have been refused but passed counts against it.
- **Gate recall:** of the answerable questions, the share that passed the gate.
- **Correctly refused:** of the questions that should be refused, the share that did not pass the gate.

### Results

| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |
|---|---|---|---|---|---|---|
| full (default) | 100% (79/79) | 99% (78/79) | 84% (66/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| no keyword search | 99% (78/79) | 95% (75/79) | 82% (65/79) | 99% (78/79) | 99% (78/79) | 95% (19/20) |
| no reranker | 99% (78/79) | 91% (72/79) | 59% (47/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| vector search only | 96% (76/79) | 76% (60/79) | 54% (43/79) | 96% (75/78) | 95% (75/79) | 85% (17/20) |
| chunks of 500 characters | 100% (79/79) | 99% (78/79) | 82% (65/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |
| chunks of 2000 characters | 100% (79/79) | 99% (78/79) | 89% (70/79) | 99% (79/80) | 100% (79/79) | 95% (19/20) |

By question type (default variant): heading questions are 64/64 for the document and the page, and the hand-written questions are 15/15 for the document and 14/15 for the page. The raw numbers are in `benchmark/results.json`.

### Reading the table

- **Each part earns its place at page level.** Vector search alone finds the right page in the top 8 for 76 percent of questions, adding keyword search brings it to 91 percent, and adding the reranker gets 99 percent.
- **The reranker is what puts the right page first.** 59 percent without it, 84 percent with it.
- **Chunk size barely matters here.** The three sizes differ by a few questions, which is noise on 79 questions. The 2000-character variant looks better at "right page first" (89 percent), but the sample is too small to conclude that.
- **Finding the right document is not a hard test.** With only 9 documents nearly every variant gets 96 to 100 percent.

### The relevance gate change

The first run showed that the gate let 4 of the 6 near-miss questions through, because the nearest passages looked similar enough to pass.

**The change.** If the question contains a specific word that appears nowhere in the indexed documents (an acronym such as "GDPR", or a rare word such as "Kerberos"), the question passes only if the best reranker score is at least 3. Everyday words never trigger it, simple word endings are ignored, and the word list is read from whatever is indexed, so it follows the documents when files are added or removed.

**How it was tested without fooling myself.** The setting was chosen on a separate set of 95 questions, written afterwards for this purpose. The 99 test questions were not used for tuning.

| | Gate precision | Gate recall | Near misses refused |
|---|---|---|---|
| Tuning set, before | 94% (64/68) | 98% (64/65) | 21/25 |
| Tuning set, after | 97% (64/66) | 98% (64/65) | 23/25 |
| Test set, before | 95% (79/83) | 100% (79/79) | 2/6 |
| Test set, after | 99% (79/80) | 100% (79/79) | 5/6 |

**Two things that went wrong on the way.** A stricter first version refused the question "What is the minimum length of a password that a user picks themselves?", and it also refused "What is the minimum password length?" on the sample documents. The fix was to count only specific words. The second problem was a coding slip (two invisible control characters in a pattern), found and fixed while checking.

**Remaining weaknesses.**

- The one test question that still gets through asks which encryption algorithm NIST recommends for TLS 1.3. The documents do mention TLS, so it is arguably a fair near miss.
- A good question that uses a rare word the documents never use, such as a synonym or a misspelling, can be refused unless the best match is very strong.
- The threshold came from these 9 documents. Very different collections may need it adjusted (`UNKNOWN_TERM_RERANK_MIN` in `backend/app/config.py`).

### What the public benchmark does not show

- **No answers are graded.** It stops at retrieval and the gate. It does not check what the language model writes, whether it quotes correctly, or whether it ignores a wrong passage.
- **The sample is small.** 79 answerable and 20 refusal questions. A difference of one to four questions is noise, and no confidence intervals are given.
- **64 of the 79 answerable questions are the easy heading kind.** The 15 hand-written ones are a more honest test.
- **Indexing is not exactly repeatable.** The vector index uses approximate search, so two builds of the same documents differ slightly. Across three builds, the vector-only variant moved between 73 and 77 percent for the right page in the top 8.
- **Only the retrieval step is measured**, not the rest of the question handling (named documents, ambiguity prompts, listings).

## 2. The earlier private run (not reproducible)

An earlier version of this pipeline was run on a private collection of more than 1,000 real-world PDF documents (roughly 19,000 pages, mostly Indonesian, mixed layouts). That data cannot be shared, so these results cannot be reproduced from this repository. They show how the design behaved at scale, with less weight than the public benchmark.

In the final run, 820 questions were scored as answer-or-refuse decisions: 700 had an answer in the documents, and 120 were off-topic or made up and should have been refused. The other questions in the run (topic search across documents, listing all items, and bare identifiers that match many documents) were scored with their own measures and are not part of these figures.

| Measure | Value | How it is calculated |
|---|---|---|
| Precision | 99.2 percent | 612 of the 617 questions the system answered should have been answered |
| Recall | 87.4 percent | 612 of the 700 answerable questions were answered |
| Stability | 0 errors | All 1,000 questions completed with no runtime errors |

- **Precision** measures whether the system answered the right questions, not whether each answer was correct. The refusal set in that run was easy to spot, which flatters it.
- **Recall** counts every refusal of an answerable question as a miss. Most of the 88 misses were cases where the section had no real text to quote, so a refusal was arguably correct. That number was not recalculated, so I do not claim a higher one.
- **Not comparable with the public benchmark.** Different documents, different questions, and a harder refusal set in the public one. The public figures are the ones to quote because anyone can rerun them.

## 3. The Docker setup

The Docker Compose setup is built and run in CI on every push: the containers start, the sample documents are indexed, the website serves its page, and one question goes through the whole chain to an answer with a source. The CI run uses a tiny model, so it proves that the setup works, not that answers are good. It has not been run with the full 9B model on a local PC yet.

## 4. Glossary

| Term | Meaning |
|---|---|
| Passage (chunk) | A piece of a document, about 1,000 characters, with its page number |
| Vector search | Finds passages whose meaning is close to the question |
| Keyword search | Finds passages that share the question's words, which matters for exact terms |
| Reranker | A second model that rereads the candidate passages and puts the best first |
| Relevance gate | The check that refuses a question before the language model is asked, when nothing retrieved is relevant enough |
| Near miss | A question on a related topic that the documents do not actually cover |

## 5. Reproducing and extending

- Rerun the public benchmark: `python benchmark/run_benchmark.py`. It takes well under an hour on a laptop CPU, mostly indexing.
- Add questions in `benchmark/questions_manual.json` (answerable ones list phrases that must appear on the right page) and run it again.
- Unit tests for the gate and for the benchmark files run in CI with the rest of the backend tests.
