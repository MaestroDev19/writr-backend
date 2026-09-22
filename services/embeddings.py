import asyncio
from functools import lru_cache
from typing import Annotated, Protocol, runtime_checkable

from fastapi import Depends
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from core.config import SettingsDep, get_settings
from utils.log import logger


class EmbeddingError(Exception):
    """Raised when an embedding provider fails to embed text or initialize."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients.
    
    Supports standard LangChain embeddings clients providing synchronous
    `embed_query` / `embed_documents` and optional asynchronous `aembed_query` /
    `aembed_documents`.
    """

    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingService:
    """Service wrapper for generating query and document embeddings.
    
    Executes native async embedding if supported by the underlying client;
    otherwise safely delegates synchronous embedding calls to a worker thread
    via `asyncio.to_thread` to prevent blocking the event loop.
    """

    def __init__(self, client: EmbeddingClient) -> None:
        self.client = client
        logger.info(
            f"EmbeddingService initialized with client '{type(client).__name__}'"
        )

    async def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Query text cannot be empty.")

        logger.info("Embedding a user query.")
        try:
            if hasattr(self.client, "aembed_query") and callable(self.client.aembed_query):
                return await self.client.aembed_query(text)
            return await asyncio.to_thread(self.client.embed_query, text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise EmbeddingError(f"Failed to embed query: {e}") from e

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []

        logger.info(f"Embedding {len(documents)} document(s).")
        try:
            if hasattr(self.client, "aembed_documents") and callable(self.client.aembed_documents):
                return await self.client.aembed_documents(documents)
            return await asyncio.to_thread(self.client.embed_documents, documents)
        except Exception as e:
            logger.error(f"Error embedding {len(documents)} document(s): {e}")
            raise EmbeddingError(f"Failed to embed documents: {e}") from e


class GeminiEmbeddingService(EmbeddingService):
    """Embedding service using Google Generative AI (Gemini)."""

    def __init__(self, model: str | None = None, dimensions: int | None = None) -> None:
        settings = get_settings()
        api_key = settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key is missing in Settings.")
            raise EmbeddingError("Gemini API key is not configured in Settings.")

        target_model = model or settings.gemini_embedding_model
        dim = dimensions or settings.embedding_dim

        try:
            client = GoogleGenerativeAIEmbeddings(
                model=target_model,
                google_api_key=api_key,
                output_dimensionality=dim,
            )
        except Exception as e:
            logger.error(f"Failed to initialize GeminiEmbeddingService: {e}")
            raise EmbeddingError(f"Could not initialize GeminiEmbeddingService: {e}") from e

        super().__init__(client=client)
        logger.info("GeminiEmbeddingService initialized.")


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI."""

    def __init__(self, model: str | None = None, dimensions: int | None = None) -> None:
        settings = get_settings()
        api_key = settings.openai_api_key
        if not api_key:
            logger.error("OpenAI API key is missing in Settings.")
            raise EmbeddingError("OpenAI API key is not configured in Settings.")

        target_model = model or settings.openai_embedding_model
        dim = dimensions or settings.embedding_dim

        try:
            client = OpenAIEmbeddings(
                model=target_model,
                api_key=api_key,
                dimensions=dim,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIEmbeddingService: {e}")
            raise EmbeddingError(f"Could not initialize OpenAIEmbeddingService: {e}") from e

        super().__init__(client=client)
        logger.info("OpenAIEmbeddingService initialized.")


@lru_cache(maxsize=1)
def get_gemini_embedding_service() -> GeminiEmbeddingService:
    """Cached factory for GeminiEmbeddingService."""
    return GeminiEmbeddingService()


@lru_cache(maxsize=1)
def get_openai_embedding_service() -> OpenAIEmbeddingService:
    """Cached factory for OpenAIEmbeddingService."""
    return OpenAIEmbeddingService()


def get_embedding_service(settings: SettingsDep) -> EmbeddingService:
    """FastAPI dependency to retrieve the configured default EmbeddingService.

    Prefers Gemini if `gemini_api_key` is configured, otherwise falls back to OpenAI.
    """
    if settings.gemini_api_key:
        return get_gemini_embedding_service()
    if settings.openai_api_key:
        return get_openai_embedding_service()
    return get_gemini_embedding_service()


# Dependency injection type aliases per FastAPI best practices
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
GeminiEmbeddingServiceDep = Annotated[GeminiEmbeddingService, Depends(get_gemini_embedding_service)]
OpenAIEmbeddingServiceDep = Annotated[OpenAIEmbeddingService, Depends(get_openai_embedding_service)]


async def embed_query(text: str) -> list[float]:
    """Convenience helper to embed a single query with the default service."""
    service = get_embedding_service(get_settings())
    return await service.embed_query(text)


async def embed_documents(documents: list[str]) -> list[list[float]]:
    """Convenience helper to embed multiple documents with the default service."""
    service = get_embedding_service(get_settings())
    return await service.embed_documents(documents)