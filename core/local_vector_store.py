
from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from itertools import cycle
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
)
from uuid import uuid4
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
class LocalVectorStore(VectorStore):
    def __init__(self, client,tableName: str, embedding,index_path: str | Path = "vectors.usearch",ndim: int = 768,metric: str = "cos"):
        self.tableName = tableName
        self.embedding = embedding
        self.client = client
        self.index_path = index_path
        self.ndim = ndim
        self.index = Index(ndim=self.ndim, metric=metric)
        if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
            self.index.load(self.index_path)
    
    def add_texts(self, texts: Iterable[str],metadatas: list[dict[str, Any]] | None = None,ids: list[str] | None = None,**kwargs: Any):
        if ids is not None:
            kwargs["ids"] = ids
        texts_:Sequence[str] =  (texts if isinstance(texts, (list, tuple)) else list(texts))
        if metadatas and len(metadatas) != len(texts_):
            logger.error(f"The number of metadatas must match the number of texts.Got {len(metadatas)} metadatas and {len(texts_)} texts.")
            raise ValueError("metadatas must be a list of the same length as texts")
        
        metadatas_:list[dict[str, Any]] = [{} for _ in range(len(texts_))]
        ids_: Iterator[str | None] = iter(ids) if ids else cycle([None]) 
         
        embeddings = self.embedding.embed_texts(texts_)
        
        for i, embedding in enumerate(embeddings):
            metadata = metadatas_[i] if i < len(metadatas_) else {}
            metadata["chunk_index"] = i
            metadata["text_content"] = texts_[i]
            id = next(ids_) or str(uuid4())
            self.index.add(ids=[id], vectors=[embedding.tolist()])
            chunk=  ReferenceChunk(id=id, embedding=embedding.tobytes(), metadata=metadata)
            self.client.add(chunk)
            logger.info(f"Added chunk {id}")
        self.client.commit()
        self.index.save(self.index_path)
        logger.info(f"Saved index to {self.index_path}")
        return ids_

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if ids is not None:
            kwargs["ids"] = ids
        texts_: Sequence[str] = (texts if isinstance(texts, (list, tuple)) else list(texts))
        if metadatas and len(metadatas) != len(texts_):
            logger.error(f"The number of metadatas must match the number of texts. Got {len(metadatas)} metadatas and {len(texts_)} texts.")
            raise ValueError("metadatas must be a list of the same length as texts")

        metadatas_: list[dict[str, Any]] = [{} for _ in range(len(texts_))]
        ids_: Iterator[str | None] = iter(ids) if ids else cycle([None])

        if hasattr(self.embedding, "aembed_documents"):
            embeddings = await self.embedding.aembed_documents(list(texts_))
        elif hasattr(self.embedding, "aembed_texts"):
            embeddings = await self.embedding.aembed_texts(texts_)
        else:
            embeddings = await run_in_executor(None, self.embedding.embed_texts, texts_)

        added_ids: list[str] = []
        for i, embedding in enumerate(embeddings):
            metadata = metadatas_[i] if i < len(metadatas_) else {}
            metadata["chunk_index"] = i
            metadata["text_content"] = texts_[i]
            id = next(ids_) or str(uuid4())
            self.index.add(ids=[id], vectors=[embedding.tolist()])
            chunk = ReferenceChunk(id=id, embedding=embedding.tobytes(), metadata=metadata)
            self.client.add(chunk)
            logger.info(f"Added chunk {id}")
            added_ids.append(id)

        self.client.commit()
        self.index.save(self.index_path)
        logger.info(f"Saved index to {self.index_path}")
        return added_ids