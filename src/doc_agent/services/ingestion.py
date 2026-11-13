from collections.abc import Callable
from pathlib import Path

from doc_agent.services.base import VectorStore


def _load_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8 text") from error


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:
        raise ValueError(f"cannot read PDF {path.name}: {error}") from error


def _load_docx(path: Path) -> str:
    from docx import Document

    try:
        document = Document(str(path))
    except Exception as error:
        raise ValueError(f"cannot read DOCX {path.name}: {error}") from error
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _load_plain,
    ".md": _load_plain,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
}


def load_text(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        supported = ", ".join(sorted(LOADERS))
        raise ValueError(f"unsupported file type '{path.suffix}' (supported: {supported})")
    if not path.is_file():
        raise ValueError(f"file not found: {path}")
    text = loader(path)
    if not text.strip():
        raise ValueError(f"no extractable text in {path.name} (scanned documents need OCR)")
    return text


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
    text = text.strip()
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def ingest_file(
    path: Path, store: VectorStore, chunk_size: int, overlap: int, *, source: str | None = None
) -> int:
    chunks = split_text(load_text(path), chunk_size, overlap)
    store.add(source or path.name, chunks)  # the store assigns positions
    return len(chunks)
