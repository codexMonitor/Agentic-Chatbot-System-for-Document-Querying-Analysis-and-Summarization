from pathlib import Path

from doc_agent.services.base import VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_text(path: Path) -> str:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"unsupported file type '{path.suffix}' (supported: {supported})")
    return path.read_text(encoding="utf-8")


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


def ingest_file(path: Path, store: VectorStore, chunk_size: int, overlap: int) -> int:
    chunks = split_text(load_text(path), chunk_size, overlap)
    store.add(path.name, chunks)
    return len(chunks)
