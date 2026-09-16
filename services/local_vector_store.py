
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

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever, LangSmithRetrieverParams
from langchain_core.runnables.config import run_in_executor
class LocalVectorStore(ABC):
    def __init__(self, client,tableName: str, embedding):
        self.tableName = tableName
        self.embedding = embedding
        self.client = client
        
    
    def add_documents(self, documents: list[str]):
        pass
        

    