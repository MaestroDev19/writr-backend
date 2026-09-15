from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, LargeBinary
import uuid
class ReferenceChunks(SQLModel, table=True):
    __tablename__ = "reference_chunks"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    content: str
    metadata_json: Dict[str, Any] = Field(default={}, sa_column=Column("metadata", JSON))
    embedding: bytes = Field(sa_column=Column("embedding", LargeBinary))

