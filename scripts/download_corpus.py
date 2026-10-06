from __future__ import annotations

import hashlib
import json
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PYTHON_URL = "https://docs.python.org/3.11/archives/python-3.11.17-docs-html.tar.bz2"
NIST_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
USER_AGENT = "rag-with-evals/0.1 (+https://github.com/DanyVX/rag-with-evals)"


def download(url: str, target: Path, *, retries: int = 3) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
            return digest.hexdigest()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if target.exists():
                target.unlink()
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


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
