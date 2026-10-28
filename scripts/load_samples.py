"""Copy the sample PDFs into the backend corpus folder and build the index.

Usage (from the repository root):  python scripts/load_samples.py
Use --no-sync to only copy the files.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
BACKEND = ROOT / "backend"
CORPUS = BACKEND / "data" / "corpus"


def main() -> int:
    pdfs = sorted(SAMPLES.glob("*.pdf"))
    if not pdfs:
        print("No sample PDFs found in", SAMPLES)
        return 1
    CORPUS.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        shutil.copy2(pdf, CORPUS / pdf.name)
    print(f"Copied {len(pdfs)} sample documents to {CORPUS}")
    if "--no-sync" in sys.argv:
        return 0
    print("Building the index (the first run downloads the embedding model)...")
    return subprocess.call([sys.executable, "-m", "app.sync_documents"], cwd=BACKEND)


if __name__ == "__main__":
    sys.exit(main())
