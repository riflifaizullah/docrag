"""The bundled sample PDFs must stay parseable and keep the structure the demos rely on.

Uses only the PDF text layer, so it needs no models, no Ollama and no network.
"""
from pathlib import Path

import pytest

from app import ingestion

SAMPLES = sorted((Path(__file__).resolve().parents[2] / "samples").glob("*.pdf"))


def test_samples_exist():
    assert len(SAMPLES) >= 5


@pytest.mark.parametrize("pdf", SAMPLES, ids=lambda p: p.name)
def test_sample_has_text_pages_and_headings(pdf):
    pages = ingestion.extract_pdf_pages(pdf)
    assert len(pages) >= 2
    assert all(page.full_text.strip() for page in pages)
    headings = [h["text"] for h in ingestion.detect_headings(pages)]
    assert any("Pasal" in h for h in headings)


@pytest.mark.parametrize("pdf", SAMPLES, ids=lambda p: p.name)
def test_every_sample_has_a_lampiran_1(pdf):
    # The "which document do you mean?" demo depends on this being shared.
    pages = ingestion.extract_pdf_pages(pdf)
    headings = [h["text"] for h in ingestion.detect_headings(pages)]
    assert any(h.startswith("Lampiran 1") for h in headings)
