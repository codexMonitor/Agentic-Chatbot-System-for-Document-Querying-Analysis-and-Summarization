import re

from doc_agent.services.base import Chunk, Embedder
from doc_agent.services.errors import IndexMismatchError

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

    def delete(self, source: str) -> int:
        before = len(self._items)
        self._items = [item for item in self._items if item[0].source != source]
        return before - len(self._items)

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

    def sources(self) -> list[str]:
        return sorted({chunk.source for chunk, _ in self._items})


class ChromaVectorStore:
    """Persistent Chroma store. Vectors come from an injected `Embedder`.

    `embedder_id` identifies the embedding model; it is recorded in the collection metadata so
    vectors from different models are never mixed.
    """

    COLLECTION = "doc_agent"

    def __init__(self, persist_dir: str, embedder: Embedder, embedder_id: str) -> None:
        import chromadb

        self._embedder = embedder
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            self.COLLECTION, metadata={"hnsw:space": "cosine", "embedder": embedder_id}
        )
        recorded = (self._collection.metadata or {}).get("embedder")
        if recorded != embedder_id:
            raise IndexMismatchError(
                f"The vector store in '{persist_dir}' was built with embedder '{recorded}', but "
                f"'{embedder_id}' is configured. The stored vectors are incompatible: re-ingest "
                f"your documents after clearing '{persist_dir}', or restore the previous "
                "DOC_AGENT_EMBEDDING_PROVIDER / DOC_AGENT_EMBEDDING_MODEL."
            )

    def add(self, source: str, chunks: list[str]) -> None:
        # Re-adding a source replaces its previous chunks.
        self._collection.delete(where={"source": source})
        if not chunks:
            return
        self._collection.add(
            ids=[f"{source}#{position}" for position in range(len(chunks))],
            documents=chunks,
            embeddings=self._embedder.embed(chunks),
            metadatas=[{"source": source, "position": position} for position in range(len(chunks))],
        )

    def delete(self, source: str) -> int:
        ids = self._collection.get(where={"source": source}, include=["metadatas"])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        count = self._collection.count()
        if count == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=min(k, count),
            include=["documents", "metadatas"],
        )
        return [
            Chunk(text, str(meta["source"]), int(meta["position"]))
            for text, meta in zip(result["documents"][0], result["metadatas"][0], strict=True)
        ]

    def all_chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks = [
            Chunk(text, str(meta["source"]), int(meta["position"]))
            for text, meta in zip(result["documents"], result["metadatas"], strict=True)
        ]
        return sorted(chunks, key=lambda chunk: (chunk.source, chunk.position))

    def sources(self) -> list[str]:
        result = self._collection.get(include=["metadatas"])
        return sorted({str(metadata["source"]) for metadata in result["metadatas"]})
