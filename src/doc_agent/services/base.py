from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str  # file name
    position: int  # 0-based index of the chunk within its source


class LLMService(Protocol):
    def invoke(self, prompt: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def add(self, source: str, chunks: list[str]) -> None: ...

    def delete(self, source: str) -> int: ...

    def search(self, query: str, k: int = 4) -> list[Chunk]: ...

    def sources(self) -> list[str]: ...

    def all_chunks(self) -> list[Chunk]: ...
