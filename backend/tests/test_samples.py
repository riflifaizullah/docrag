"""The bundled sample PDFs must stay parseable and keep the structure the demos rely on."""
from pathlib import Path

import pytest

from app import ingestion, lang

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
ARTICLE_WORD = {"en": "Article", "id": "Pasal"}
APPENDIX_WORD = {"en": "Appendix", "id": "Lampiran"}
CASES = [(code, pdf) for code in ("en", "id") for pdf in sorted((SAMPLES_DIR / code).glob("*.pdf"))]
CASE_IDS = [f"{code}-{pdf.name}" for code, pdf in CASES]


def _headings(code, pdf):
    lang.set_language(code)
    pages = ingestion.extract_pdf_pages(pdf)
    return pages, [h["text"] for h in ingestion.detect_headings(pages)]


@pytest.mark.parametrize("code", ["en", "id"])
def test_samples_exist(code):
    assert len(list((SAMPLES_DIR / code).glob("*.pdf"))) >= 5


@pytest.mark.parametrize("code,pdf", CASES, ids=CASE_IDS)
def test_sample_has_text_pages_and_headings(code, pdf):
    pages, headings = _headings(code, pdf)
    assert len(pages) >= 2
    assert all(page.full_text.strip() for page in pages)
    assert any(ARTICLE_WORD[code] in h for h in headings)


@pytest.mark.parametrize("code,pdf", CASES, ids=CASE_IDS)
def test_every_sample_has_appendix_1(code, pdf):
    # The "which document do you mean?" demo depends on this being shared.
    _, headings = _headings(code, pdf)
    assert any(h.startswith(f"{APPENDIX_WORD[code]} 1") for h in headings)
