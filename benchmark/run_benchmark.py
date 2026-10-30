"""Retrieval benchmark on public documents.

    python benchmark/fetch_documents.py                      # once
    python benchmark/run_benchmark.py [--docs benchmark/documents] [--work benchmark/.work]

It indexes the documents (once per chunk size), then asks every question through the real retrieval code
(vector plus keyword search, reranker, relevance gate) and compares variants. No language model is called, so a
full run takes minutes and gives the same numbers every time. Results are written to benchmark/results.md.

What is measured, and what is not: see benchmark/README.md.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
BACKEND = HERE.parent / "backend"
CHUNK_SIZES = (500, 1000, 2000)

# name -> (chunk size, hybrid search, reranker)
VARIANTS = {
    "full (default)": (1000, True, True),
    "no keyword search": (1000, False, True),
    "no reranker": (1000, True, False),
    "vector search only": (1000, False, False),
    "chunks of 500 characters": (500, True, True),
    "chunks of 2000 characters": (2000, True, True),
}


def normalise(text: str) -> str:
    kept = [line for line in text.splitlines() if "...." not in line]  # drop table-of-contents lines
    return re.sub(r"\s+", " ", " ".join(kept)).lower()


def accepted_pages(docs_dir: Path, document: str, groups: list[list[str]]) -> list[int]:
    import pdfplumber

    pages = []
    with pdfplumber.open(docs_dir / document) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = normalise(page.extract_text() or "")
            if any(all(phrase in text for phrase in group) for group in groups):
                pages.append(number)
    return pages


def load_questions(docs_dir: Path) -> dict:
    manual = json.loads((HERE / "questions_manual.json").read_text(encoding="utf-8"))
    headings = json.loads((HERE / "questions_headings.json").read_text(encoding="utf-8"))
    answerable = [{**q, "category": "heading"} for q in headings]
    for q in manual["answerable"]:
        pages = accepted_pages(docs_dir, q["document"], q["groups"])
        if not pages:
            sys.exit(f"{q['id']}: no page contains the expected text, check the document version")
        answerable.append({"id": q["id"], "question": q["question"], "document": q["document"], "pages": pages, "category": "hand-written"})
    return {"answerable": answerable, "refuse": manual["should_refuse"]}


def evaluate(variant: str, questions: dict) -> dict:
    """Runs inside a worker process with DOCRAG_DATA_DIR and DOCRAG_CHUNK_SIZE already set."""
    sys.path.insert(0, str(BACKEND))
    from app import config, generation, retrieval

    _, config.HYBRID_SEARCH_ENABLED, config.RERANK_ENABLED = VARIANTS[variant]

    def ask(question: str) -> tuple[list[dict], bool]:
        hits = retrieval.retrieve(question)
        return hits, generation._passes_relevance_gate(hits)

    rows = {"heading": [], "hand-written": []}
    tp = fn = fp = tn = 0
    for q in questions["answerable"]:
        hits, passed = ask(q["question"])

        def right_page(h):
            return h["source"] == q["document"] and h["page"] in q["pages"]

        rows[q["category"]].append({
            "doc_top1": bool(hits) and hits[0]["source"] == q["document"],
            "doc_hit": any(h["source"] == q["document"] for h in hits),
            "page_top1": bool(hits) and right_page(hits[0]),
            "page_hit": any(right_page(h) for h in hits),
        })
        tp += passed
        fn += not passed
    answered_by_mistake = []
    for q in questions["refuse"]:
        _, passed = ask(q["question"])
        fp += passed
        tn += not passed
        if passed:
            answered_by_mistake.append(q["id"])
    return {"variant": variant, "rows": rows, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "answered_by_mistake": answered_by_mistake}


def run_worker(variant: str, data_dir: Path, docs_dir: Path) -> dict:
    env = {**os.environ, "DOCRAG_DATA_DIR": str(data_dir), "DOCRAG_CHUNK_SIZE": str(VARIANTS[variant][0])}
    out = subprocess.run(
        [sys.executable, str(Path(__file__)), "--worker", variant, "--docs", str(docs_dir)],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def index(data_dir: Path, docs_dir: Path, chunk_size: int) -> None:
    corpus = data_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for pdf in docs_dir.glob("*.pdf"):
        target = corpus / pdf.name
        if not target.exists():
            target.write_bytes(pdf.read_bytes())
    env = {**os.environ, "DOCRAG_DATA_DIR": str(data_dir), "DOCRAG_CHUNK_SIZE": str(chunk_size), "DOCRAG_VLM": "0"}
    print(f"indexing with chunks of {chunk_size} characters ...", flush=True)
    subprocess.run([sys.executable, "-m", "app.sync_documents"], cwd=BACKEND, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def rate(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}% ({k}/{n})" if n else "n/a"


def report(results: list[dict]) -> str:
    lines = ["| Variant | Right document in top 8 | Right page in top 8 | Right page first | Gate precision | Gate recall | Correctly refused |",
             "|---|---|---|---|---|---|---|"]
    for res in results:
        rows = res["rows"]["heading"] + res["rows"]["hand-written"]
        n = len(rows)
        tp, fp, fn, tn = res["tp"], res["fp"], res["fn"], res["tn"]
        lines.append("| " + " | ".join([
            res["variant"],
            rate(sum(r["doc_hit"] for r in rows), n),
            rate(sum(r["page_hit"] for r in rows), n),
            rate(sum(r["page_top1"] for r in rows), n),
            rate(tp, tp + fp), rate(tp, tp + fn), rate(tn, tn + fp),
        ]) + " |")
    full = results[0]
    lines += ["", "By question type, default variant:", "",
              "| Question type | Right document in top 8 | Right page in top 8 |", "|---|---|---|"]
    for cat in ("heading", "hand-written"):
        rows = full["rows"][cat]
        lines.append(f"| {cat} | {rate(sum(r['doc_hit'] for r in rows), len(rows))} | {rate(sum(r['page_hit'] for r in rows), len(rows))} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=HERE / "documents")
    parser.add_argument("--work", type=Path, default=HERE / ".work")
    parser.add_argument("--worker")
    args = parser.parse_args()
    questions = load_questions(args.docs)

    if args.worker:
        print(json.dumps(evaluate(args.worker, questions)))
        return

    for size in CHUNK_SIZES:
        index(args.work / f"chunks_{size}", args.docs, size)
    results = []
    for variant, (size, _, _) in VARIANTS.items():
        print(f"asking {len(questions['answerable']) + len(questions['refuse'])} questions: {variant}", flush=True)
        results.append(run_worker(variant, args.work / f"chunks_{size}", args.docs))
    (HERE / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    table = report(results)
    (HERE / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
