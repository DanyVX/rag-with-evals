from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

PYTHON_URL = "https://docs.python.org/3.11/archives/python-3.11.17-docs-html.tar.bz2"
NIST_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"


def download(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as out:
        digest = hashlib.sha256()
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path("data/raw")
    python_archive = root / "python-3.11.17-docs-html.tar.bz2"
    nist_pdf = root / "NIST.AI.100-1.pdf"

    checksums = {
        str(python_archive): {
            "url": PYTHON_URL,
            "sha256": download(PYTHON_URL, python_archive),
        },
        str(nist_pdf): {
            "url": NIST_URL,
            "sha256": download(NIST_URL, nist_pdf),
        },
    }

    extracted = root / "python-3.11.17-docs-html"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(python_archive, "r:bz2") as archive:
        archive.extractall(extracted, filter="data")

    Path("corpus").mkdir(exist_ok=True)
    Path("corpus/checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("Downloaded corpora and wrote corpus/checksums.json")


if __name__ == "__main__":
    main()
