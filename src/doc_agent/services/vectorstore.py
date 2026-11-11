import re

from doc_agent.services.base import Chunk

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "the a an and or of to in is are what does do how about for on with it this that".split()
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


class InMemoryVectorStore:
    """Keyword-overlap store for tests and offline demos."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, set[str]]] = []

    def add(self, source: str, chunks: list[str]) -> None:
        # Re-adding a source replaces its previous chunks.
        self._items = [item for item in self._items if item[0].source != source]
        for position, text in enumerate(chunks):
            self._items.append((Chunk(text, source, position), _tokens(text)))

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        query_tokens = _tokens(query)
        scored = [
            (len(query_tokens & tokens), index, chunk)
            for index, (chunk, tokens) in enumerate(self._items)
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk for _, _, chunk in scored[:k]]

    def all_chunks(self) -> list[Chunk]:
        return [chunk for chunk, _ in self._items]
