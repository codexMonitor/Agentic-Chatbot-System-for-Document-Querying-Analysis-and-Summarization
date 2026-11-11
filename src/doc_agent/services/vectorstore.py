import re

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    "the a an and or of to in is are what does do how about for on with it this that".split()
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


class InMemoryVectorStore:
    """Keyword-overlap store for tests and offline demos."""

    def __init__(self) -> None:
        self._items: list[tuple[str, set[str]]] = []

    def add(self, doc_id: str, chunks: list[str]) -> None:
        for chunk in chunks:
            self._items.append((chunk, _tokens(chunk)))

    def search(self, query: str, k: int = 4) -> list[str]:
        query_tokens = _tokens(query)
        scored = [
            (len(query_tokens & tokens), index, text)
            for index, (text, tokens) in enumerate(self._items)
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [text for _, _, text in scored[:k]]

    def all_chunks(self) -> list[str]:
        return [text for text, _ in self._items]
