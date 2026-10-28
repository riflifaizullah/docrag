"""Copy the sample PDFs for one language into the backend corpus folder and build the index.

Usage (from the repository root):
    python scripts/load_samples.py [en|id] [--no-sync]

English is the default. Run the backend with the same DOCRAG_LANG so prompts and headings match.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    lang = args[0] if args else "en"
    samples = ROOT / "samples" / lang
    pdfs = sorted(samples.glob("*.pdf"))
    if not pdfs:
        print("No sample PDFs found in", samples)
        return 1
    data_dir = Path(os.getenv("DOCRAG_DATA_DIR", str(BACKEND / "data")))
    corpus = data_dir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        shutil.copy2(pdf, corpus / pdf.name)
    print(f"Copied {len(pdfs)} {lang} sample documents to {corpus}")
    if "--no-sync" in sys.argv:
        return 0
    print("Building the index (the first run downloads the embedding model)...")
    env = {**os.environ, "DOCRAG_LANG": lang}
    return subprocess.call([sys.executable, "-m", "app.sync_documents"], cwd=BACKEND, env=env)


if __name__ == "__main__":
    sys.exit(main())
