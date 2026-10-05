"""Format-specific loaders. Unsupported or unsafe input is rejected clearly."""

from pathlib import Path

from bs4 import BeautifulSoup


class UnsupportedDocumentError(ValueError):
    """Raised for a file type with no reliable configured loader."""


def load_text(path: Path) -> tuple[str, dict[str, str | int]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst"}:
        return path.read_text(encoding="utf-8", errors="replace"), {"format": suffix.lstrip(".")}
    if suffix in {".html", ".htm"}:
        source = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(source, "html.parser")
        for unwanted in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            unwanted.decompose()
        text = soup.get_text("\n", strip=True)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        return text, {"format": "html", "title": title}
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found,unused-ignore]
        except ImportError as error:
            raise UnsupportedDocumentError(
                "PDF support requires the 'pdf' optional dependency"
            ) from error
        pdf = fitz.open(path)
        pages = [page.get_text("text") for page in pdf]
        return "\n\n".join(pages), {"format": "pdf", "pages": len(pages)}
    raise UnsupportedDocumentError(f"Unsupported document suffix: {suffix or '<none>'}")
