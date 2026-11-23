"""Document loading: PDF/text extraction and overlapping chunking."""

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from summarizer import SummarizerError

SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def file_hash(path) -> str:
    """SHA-256 of the file content (used as the embedding-cache key)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def extract_pages(path) -> list[tuple]:
    """Return [(page_number, text), ...]. Page number is None for text files."""
    path = Path(path)
    if not path.exists():
        raise SummarizerError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise SummarizerError("Unsupported file type. Use .pdf, .txt or .md files.")

    try:
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise SummarizerError(f"{path.name} is password-protected.")
            pages = [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        else:
            pages = [(None, path.read_text(encoding="utf-8", errors="replace"))]
    except SummarizerError:
        raise
    except Exception as exc:
        raise SummarizerError(f"Could not read {path.name}: {exc}")

    pages = [(n, _clean(t)) for n, t in pages if t.strip()]
    if not pages:
        raise SummarizerError(
            f"No text found in {path.name}. Scanned or image-only PDFs need OCR, "
            "which is not supported yet."
        )
    return pages


def extract_text(pdf_path) -> str:
    """Return the full text of a PDF (or text file)."""
    return "\n\n".join(text for _, text in extract_pages(pdf_path))


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, cutting at word boundaries."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Need size > 0 and 0 <= overlap < size.")
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        space = text.find(" ", max(end - overlap, start + 1))
        start = space + 1 if 0 <= space < end else end
    return chunks


def build_chunks(pages, source: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk every page, remembering which document and page each chunk came from."""
    chunks = []
    for page, text in pages:
        for piece in chunk_text(text, size, overlap):
            chunks.append({"text": piece, "source": source, "page": page})
    return chunks
