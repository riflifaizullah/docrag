"""The benchmark's question and document files stay consistent with each other."""
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "benchmark"


def _load(name):
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def test_every_question_points_at_a_listed_document():
    documents = {d["file"] for d in _load("documents.json")}
    manual = _load("questions_manual.json")
    dev = _load("questions_dev.json")
    questions = _load("questions_headings.json") + _load("questions_dev_headings.json") + manual["answerable"] + dev["answerable"]
    assert {q["document"] for q in questions} <= documents


def test_question_ids_are_unique_and_pages_present():
    manual = _load("questions_manual.json")
    questions = _load("questions_headings.json")
    ids = [q["id"] for q in questions + manual["answerable"] + manual["should_refuse"]]
    assert len(ids) == len(set(ids))
    assert all(q["pages"] for q in questions)
    assert all(q["groups"] for q in manual["answerable"])


def test_documents_have_checksums():
    assert all(len(d["sha256"]) == 64 for d in _load("documents.json"))
