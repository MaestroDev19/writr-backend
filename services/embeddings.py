from google import genai
from google.genai import types
from fastapi import HTTPException, status

from core.config import get_settings
from utils.log import logger

_genai_client: genai.Client | None = None


def prepare_query(query: str) -> str:
    return f"task: search result | query: {query}"


def prepare_document(content: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {content}"


def get_genai_client() -> genai.Client:
    """Gemini client using GEMINI_API_KEY from Settings."""
    global _genai_client
    if _genai_client is None:
        settings = get_settings()
        if not settings.gemini_api_key:
            logger.error("Gemini API key is missing in Settings.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GEMINI_API_KEY must be set in environment variables.",
            )
        _genai_client = genai.Client(api_key=settings.gemini_api_key)
        logger.info("Gemini client initialized model=%s dim=%s", settings.embedding_model, settings.embedding_dim)
    return _genai_client


def _embedding_values(result: types.EmbedContentResponse) -> list[list[float]]:
    settings = get_settings()
    embeddings = result.embeddings or []
    vectors: list[list[float]] = []
    for item in embeddings:
        values = item.values
        if not values:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Gemini returned an empty embedding.",
            )
        if len(values) != settings.embedding_dim:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Gemini embedding dim {len(values)} does not match {settings.embedding_dim}.",
            )
        vectors.append(list(values))
    return vectors


async def embed_documents(texts: list[str], *, title: str | None = None) -> list[list[float]]:
    """Embed reference chunks with gemini-embedding-2 at Settings.embedding_dim."""
    if not texts:
        return []
    settings = get_settings()
    client = get_genai_client()
    contents = [
        types.Content(parts=[types.Part.from_text(text=prepare_document(text, title))])
        for text in texts
    ]
    result = await client.aio.models.embed_content(
        model=settings.embedding_model,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
    )
    return _embedding_values(result)


async def embed_query(query: str) -> list[float]:
    """Embed a retrieval query with the same model and dimension as stored chunks."""
    settings = get_settings()
    client = get_genai_client()
    result = await client.aio.models.embed_content(
        model=settings.embedding_model,
        contents=prepare_query(query),
        config=types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
    )
    vectors = _embedding_values(result)
    if not vectors:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned no query embedding.",
        )
    return vectors[0]
