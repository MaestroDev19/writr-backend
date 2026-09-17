from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from sqlmodel import select
from typing_extensions import override
from usearch.index import Index

from core.config import get_settings
from core.local_db import get_session
from core.local_embedding import get_local_embedding
from core.matching import match_reference_chunks
from models.local.reference_chunk import ReferenceChunks
from utils import logger


def _as_vector(embedding: Any) -> np.ndarray:
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _usearch_key(chunk_id: str) -> int:
    try:
        return UUID(chunk_id).int & ((1 << 63) - 1)
    except ValueError:
        return hash(chunk_id) & ((1 << 63) - 1)


class LocalVectorStore(VectorStore):
    """SQLite payloads + USearch index for *reference* chunks only.

    Target documents must never be added here. They are request-scoped context
    for the agent and are returned as a copiable message, not written back.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index_path: str | Path | None = None,
        ndim: int | None = None,
        metric: str = "cos",
    ) -> None:
        settings = get_settings()
        self.embedding = embedding
        self.index_path = Path(index_path or settings.usearch_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.ndim = ndim or settings.embedding_dim
        self.metric = metric
        self._key_to_id: dict[int, str] = {}
        self.index = Index(ndim=self.ndim, metric=metric)
        self._load_index()
        self._hydrate_from_db()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _load_index(self) -> None:
        if self.index_path.exists() and os.path.getsize(self.index_path) > 0:
            self.index.load(str(self.index_path))

    def _hydrate_from_db(self) -> None:
        with get_session() as session:
            chunks = list(session.exec(select(ReferenceChunks)).all())
        if not chunks:
            return

        needs_rebuild = len(self.index) == 0
        for chunk in chunks:
            key = _usearch_key(chunk.id)
            self._key_to_id[key] = chunk.id
            if needs_rebuild:
                vector = np.frombuffer(chunk.embedding, dtype=np.float32)
                if vector.size != self.ndim:
                    logger.warning(
                        "Skipping chunk %s: stored dim %s != index dim %s",
                        chunk.id,
                        vector.size,
                        self.ndim,
                    )
                    continue
                self.index.add(key, vector)
        if needs_rebuild and len(self.index) > 0:
            self.index.save(str(self.index_path))
            logger.info("Rebuilt USearch index with %s reference vectors", len(self.index))

    def _encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if hasattr(self.embedding, "embed_documents"):
            return self.embedding.embed_documents(list(texts))
        return self.embedding.embed_texts(texts)

    async def _aencode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if hasattr(self.embedding, "aembed_documents"):
            return await self.embedding.aembed_documents(list(texts))
        if hasattr(self.embedding, "aembed_texts"):
            return await self.embedding.aembed_texts(list(texts))
        return await run_in_executor(None, self._encode_documents, texts)

    def _persist(self, texts: Sequence[str], metadatas: list[dict[str, Any]], ids: list[str | None], raw_embeddings: Sequence[Any]) -> list[str]:
        added_ids: list[str] = []
        with get_session() as session:
            for i, raw in enumerate(raw_embeddings):
                vector = _as_vector(raw)
                if vector.size != self.ndim:
                    raise ValueError(
                        f"Embedding dim {vector.size} does not match store dim {self.ndim}. "
                        "Re-index references after changing OLLAMA_EMBEDDING_MODEL."
                    )
                metadata = dict(metadatas[i])
                if metadata.get("kind") == "target":
                    raise ValueError(
                        "Target documents must not be stored in the notes library. "
                        "Pass them as request-scoped context only."
                    )
                metadata.setdefault("chunk_index", i)
                metadata["kind"] = "reference"
                chunk_id = ids[i] or str(uuid4())
                key = _usearch_key(chunk_id)
                self.index.add(key, vector)
                self._key_to_id[key] = chunk_id
                session.add(
                    ReferenceChunks(
                        id=chunk_id,
                        content=texts[i],
                        metadata_json=metadata,
                        embedding=vector.tobytes(),
                    )
                )
                added_ids.append(chunk_id)
                logger.info("Added reference chunk %s", chunk_id)
            session.commit()
        self.index.save(str(self.index_path))
        logger.info("Saved index to %s", self.index_path)
        return added_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_, metadatas_, ids_ = self._normalize_inputs(texts, metadatas, ids)
        return self._persist(texts_, metadatas_, ids_, self._encode_documents(texts_))

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts_, metadatas_, ids_ = self._normalize_inputs(texts, metadatas, ids)
        raw = await self._aencode_documents(texts_)
        return self._persist(texts_, metadatas_, ids_, raw)

    def _normalize_inputs(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None,
        ids: list[str] | None,
    ) -> tuple[Sequence[str], list[dict[str, Any]], list[str | None]]:
        texts_ = texts if isinstance(texts, (list, tuple)) else list(texts)
        if metadatas is not None and len(metadatas) != len(texts_):
            raise ValueError("metadatas must be a list of the same length as texts")
        if ids is not None and len(ids) != len(texts_):
            raise ValueError("ids must be a list of the same length as texts")
        metadatas_ = [dict(item) for item in metadatas] if metadatas else [{} for _ in range(len(texts_))]
        ids_ = list(ids) if ids else [None] * len(texts_)
        return texts_, metadatas_, ids_

    @override
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    @override
    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        docs_and_scores = await self.asimilarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        query_embedding = self.embedding.embed_query(query)
        return self._docs_from_query_vector(query_embedding, k, kwargs.get("filter"))

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        if hasattr(self.embedding, "aembed_query"):
            query_embedding = await self.embedding.aembed_query(query)
        else:
            query_embedding = await run_in_executor(None, self.embedding.embed_query, query)
        return self._docs_from_query_vector(query_embedding, k, kwargs.get("filter"))

    def _docs_from_query_vector(
        self,
        query_embedding: list[float],
        k: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        if filter_dict or len(self.index) == 0:
            return self._docs_from_numpy(query_embedding, k, filter_dict)
        return self._docs_from_usearch(query_embedding, k)

    def _docs_from_usearch(self, query_embedding: list[float], k: int) -> list[tuple[Document, float]]:
        count = min(k, len(self.index))
        if count <= 0:
            return []
        vector = _as_vector(query_embedding)
        matches = self.index.search(vector, count)
        keys = [int(key) for key in getattr(matches, "keys", [])]
        distances = [float(dist) for dist in getattr(matches, "distances", [])]
        ordered_ids = [self._key_to_id[key] for key in keys if key in self._key_to_id]
        if not ordered_ids:
            return self._docs_from_numpy(query_embedding, k, None)

        with get_session() as session:
            rows = session.exec(select(ReferenceChunks).where(ReferenceChunks.id.in_(ordered_ids))).all()
        by_id = {row.id: row for row in rows}
        docs: list[tuple[Document, float]] = []
        for chunk_id, distance in zip(ordered_ids, distances, strict=False):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            similarity = 1.0 - distance
            docs.append((self._to_document(chunk, similarity), similarity))
        return docs

    def _docs_from_numpy(
        self,
        query_embedding: list[float],
        k: int,
        filter_dict: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        with get_session() as session:
            matches = match_reference_chunks(session, query_embedding, match_count=k, filter_dict=filter_dict)
        return [(self._to_document(match["chunk"], match["similarity"]), match["similarity"]) for match in matches]

    def _to_document(self, chunk: ReferenceChunks, similarity: float) -> Document:
        metadata = dict(chunk.metadata_json or {})
        metadata["chunk_id"] = chunk.id
        metadata["similarity"] = similarity
        return Document(page_content=chunk.content, metadata=metadata)

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> LocalVectorStore:
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if not ids:
            return False
        with get_session() as session:
            for chunk_id in ids:
                chunk = session.get(ReferenceChunks, chunk_id)
                if chunk is None:
                    continue
                key = _usearch_key(chunk_id)
                try:
                    self.index.remove(key)
                except Exception:
                    logger.warning("USearch key missing for chunk %s", chunk_id)
                self._key_to_id.pop(key, None)
                session.delete(chunk)
            session.commit()
        self.index.save(str(self.index_path))
        return True


@lru_cache(maxsize=1)
def get_local_vector_store() -> LocalVectorStore:
    embedding = get_local_embedding()
    ndim = getattr(embedding, "dimensions", None) or get_settings().embedding_dim
    return LocalVectorStore(embedding=embedding, ndim=ndim)
