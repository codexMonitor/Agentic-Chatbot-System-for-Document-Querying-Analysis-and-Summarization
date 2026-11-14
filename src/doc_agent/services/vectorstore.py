import os
import re
from typing import Any

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


class SupabaseVectorStore:
    """Server-side pgvector store; collection metadata guards against mixing embeddings."""

    def __init__(
        self,
        embedder: Embedder,
        embedder_id: str,
        *,
        client: Any = None,
        collection: str = "doc_agent",
    ) -> None:
        if client is None:
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_API_KEY")
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_API_KEY must be configured.")
            from supabase import create_client

            client = create_client(url, key)
        self._client = client
        self._embedder = embedder
        self._embedder_id = embedder_id
        self._collection = collection
        recorded = (
            self._client.rpc(
                "doc_agent_init_collection",
                {"collection_name": collection, "configured_embedder": embedder_id},
            )
            .execute()
            .data
        )
        if recorded != embedder_id:
            raise IndexMismatchError(
                f"The Supabase collection was built with embedder '{recorded}', but "
                f"'{embedder_id}' is configured. The stored vectors are incompatible: "
                "clear the collection and re-ingest your documents, or restore the previous "
                "DOC_AGENT_EMBEDDING_PROVIDER / DOC_AGENT_EMBEDDING_MODEL."
            )

    def add(self, source: str, chunks: list[str]) -> None:
        embeddings = self._embedder.embed(chunks) if chunks else []
        if len(embeddings) != len(chunks):
            raise ValueError("The embedder returned a different number of vectors than chunks.")
        rows = [
            {"text": text, "embedding": embedding}
            for text, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._client.rpc(
            "doc_agent_replace_source",
            {
                "collection_name": self._collection,
                "configured_embedder": self._embedder_id,
                "source_name": source,
                "chunks": rows,
            },
        ).execute()

    def delete(self, source: str) -> int:
        return int(
            self._client.rpc(
                "doc_agent_delete_source",
                {"collection_name": self._collection, "source_name": source},
            )
            .execute()
            .data
        )

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if k <= 0:
            return []
        rows = (
            self._client.rpc(
                "doc_agent_match_chunks",
                {
                    "collection_name": self._collection,
                    "query_embedding": self._embedder.embed([query])[0],
                    "match_count": k,
                },
            )
            .execute()
            .data
        )
        return [Chunk(row["text"], row["source"], row["position"]) for row in rows]

    def all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        page_size = 1000
        offset = 0
        while True:
            rows = (
                self._client.table("doc_agent_chunks")
                .select("text,source,position")
                .eq("collection", self._collection)
                .order("source")
                .order("position")
                .range(offset, offset + page_size - 1)
                .execute()
                .data
            )
            chunks.extend(Chunk(row["text"], row["source"], row["position"]) for row in rows)
            if len(rows) < page_size:
                return chunks
            offset += len(rows)

    def sources(self) -> list[str]:
        return sorted({chunk.source for chunk in self.all_chunks()})
