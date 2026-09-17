from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from core.config import get_settings

# EmbeddingGemma retrieval prefixes (Google). LangChain/Ollama do not add these.
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


class LocalEmbedding(Embeddings):
    """Pinned local embedder for the notes library.

    Uses Ollama `embeddinggemma` (768-d). Session drafts must never pass through
    embed_documents — only reference notes are indexed.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        keep_alive: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_embedding_model
        self.dimensions = dimensions or settings.embedding_dim
        self.keep_alive = keep_alive if keep_alive is not None else settings.ollama_embed_keep_alive
        self._client = OllamaEmbeddings(
            model=self.model,
            base_url=base_url or settings.ollama_endpoint,
            keep_alive=self.keep_alive,
            validate_model_on_init=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents([DOCUMENT_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(QUERY_PREFIX + text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents([DOCUMENT_PREFIX + text for text in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(QUERY_PREFIX + text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(list(texts))

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.aembed_documents(list(texts))

    def warmup(self) -> int:
        vector = self.embed_query("warmup")
        return len(vector)

    async def awarmup(self) -> int:
        vector = await self.aembed_query("warmup")
        return len(vector)


@lru_cache(maxsize=1)
def get_local_embedding() -> LocalEmbedding:
    return LocalEmbedding()
