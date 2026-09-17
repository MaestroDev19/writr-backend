from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from core.config import get_settings
from core.local_vector_store import get_local_vector_store

WRITE_PROMPT = """You are Writr's Write companion.
Search the notes library before rewriting so lore, character sheets, and style notes are respected.
The session story draft is request-scoped: never store it and never treat it as a note.
Return copiable revised text. Do not mention tools unless the author asks how you worked."""

REVIEW_PROMPT = """You are Writr's Review companion.
Search the notes library so feedback matches the author's world and voice.
The pasted document is request-scoped: never store it.
Give focused feedback (structure, flow, voice, or line polish as asked) with concrete "Try this" suggestions.
Do not rewrite the whole piece unless the author asks."""

_REVIEW_HINTS = ("review", "critique", "structure", "flow", "voice", "polish", "feedback")


class ReferenceIndex:
    """Async facade over the reference-only vector store (notes library)."""

    async def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        store = get_local_vector_store()
        return await store.aadd_texts(texts, metadatas=metadatas, ids=ids)


reference_index = ReferenceIndex()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    settings = get_settings()
    size = chunk_size or settings.reference_chunk_size
    step_overlap = settings.reference_chunk_overlap if overlap is None else overlap
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - step_overlap, start + 1)
    return chunks


def _is_review(instruction: str) -> bool:
    lowered = instruction.lower()
    return any(hint in lowered for hint in _REVIEW_HINTS)


async def run_local_agent(
    *,
    instruction: str,
    generate_model: str,
    target_text: str | None = None,
    match_count: int = 5,
) -> dict:
    settings = get_settings()
    store = get_local_vector_store()

    @tool
    def search_reference_notes(query: str) -> str:
        """Search the author's notes library (lore, character sheets, research, style).

        Use this before rewriting or reviewing. Do not search for the session draft;
        that text is already in the user message and is not stored.

        Args:
            query: Short query about characters, lore, or writing style.
        """
        docs = store.similarity_search(query, k=match_count)
        if not docs:
            return "No matching notes found."
        parts: list[str] = []
        for doc in docs:
            source = doc.metadata.get("filename", "note")
            parts.append(f"[{source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    llm = ChatOllama(
        model=generate_model,
        base_url=settings.ollama_endpoint,
        keep_alive="5m",
        temperature=0.3,
        validate_model_on_init=False,
    )
    agent = create_agent(
        model=llm,
        tools=[search_reference_notes],
        system_prompt=REVIEW_PROMPT if _is_review(instruction) else WRITE_PROMPT,
    )
    user_content = instruction.strip()
    if target_text:
        user_content = (
            f"{user_content}\n\n--- Session document (not stored in notes) ---\n{target_text}"
        )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"recursion_limit": 10},
    )
    output = result["messages"][-1].content
    if not isinstance(output, str):
        output = str(output)
    return {
        "message": output,
        "generate_model": generate_model,
        "embedding_model": settings.ollama_embedding_model,
        "target_stored": False,
    }
