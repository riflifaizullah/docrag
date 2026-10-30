"""Downloads the benchmark documents listed in documents.json and checks each file's SHA-256.

    python benchmark/fetch_documents.py [target_folder]

The PDFs are not stored in this repository. They are US government works and Indonesian laws.
"""
import hashlib
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).parent


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "documents"
    target.mkdir(parents=True, exist_ok=True)
    bad = 0
    for doc in json.loads((HERE / "documents.json").read_text(encoding="utf-8")):
        path = target / doc["file"]
        if not path.exists():
            print(f"downloading {doc['file']} ...")
            response = requests.get(doc["url"], headers={"User-Agent": "docrag-benchmark"}, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        ok = digest == doc["sha256"]
        bad += not ok
        print(f"{'ok      ' if ok else 'CHANGED '}{doc['file']}")
    if bad:
        sys.exit(f"{bad} file(s) differ from the recorded checksum. The source may have been updated, so results can differ.")


if __name__ == "__main__":
    main()
