# Corpus manifest

## Primary corpus: Python 3.11.17 documentation

Source: official Python documentation HTML archive.

Purpose: a license-clear documentation corpus large enough to produce thousands of chunks and exercise exact identifiers, code, cross-references, short queries, and multi-section questions.

License: Python documentation is distributed under the Python Software Foundation licensing terms documented by the Python project. Preserve upstream attribution and license notices when redistributing corpus material.

Pinned archive:
- python-3.11.17-docs-html.tar.bz2
- https://docs.python.org/3.11/archives/python-3.11.17-docs-html.tar.bz2

## Layout-stress corpus: NIST AI RMF 1.0

Source: NIST AI 100-1, January 2023.

Purpose: a public U.S. government PDF used to test PDF layout, headings, tables, repeated page furniture, and extraction warnings.

Pinned PDF:
- NIST.AI.100-1.pdf
- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf

## Reproducibility

Run:

    python scripts/download_corpus.py

The downloader records the SHA-256 digest of every downloaded artifact in corpus/checksums.json. That generated checksum manifest is the authoritative checksum record for an evaluation run and must be archived with published results. Raw corpus files are intentionally gitignored.
