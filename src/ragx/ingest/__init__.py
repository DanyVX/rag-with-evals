"""Document loading, normalization, and deduplication."""

from ragx.ingest.models import Document
from ragx.ingest.service import ingest_paths

__all__ = ["Document", "ingest_paths"]
