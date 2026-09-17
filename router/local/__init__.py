from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from core.config import Settings, get_settings
from core.local_agent import chunk_text, reference_index, run_local_agent
from core.local_embedding import LocalEmbedding, get_local_embedding
from router.local.schemas import (
    EmbedRequest,
    EmbedResponse,
    ModelsResponse,
    OllamaModel,
    ReferenceIn,
    ReferenceOut,
    RunRequest,
    RunResponse,
)
from utils import logger

router = APIRouter(prefix="/local", tags=["local"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingDep = Annotated[LocalEmbedding, Depends(get_local_embedding)]


def _classify_role(capabilities: list[str]) -> str:
    if "embedding" in capabilities:
        return "embedding"
    if "completion" in capabilities:
        return "generate"
    return "unknown"


@router.get("/models")
async def list_models(settings: SettingsDep) -> ModelsResponse:
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_endpoint, timeout=30.0) as client:
            tags_response = await client.get("/api/tags")
            tags_response.raise_for_status()
            tagged = tags_response.json().get("models", [])
            shows = await _show_models(client, [item["name"] for item in tagged])
    except httpx.HTTPError as exc:
        logger.error("Failed to list Ollama models: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama is unreachable. Start `ollama serve` and retry.",
        ) from exc

    models: list[OllamaModel] = []
    for item, show in zip(tagged, shows, strict=True):
        capabilities = show.get("capabilities") or []
        details = item.get("details") or {}
        models.append(
            OllamaModel(
                name=item["name"],
                role=_classify_role(capabilities),
                capabilities=capabilities,
                size=item.get("size"),
                family=details.get("family"),
            )
        )
    return ModelsResponse(
        embedding_model=settings.ollama_embedding_model,
        generate_models=[model for model in models if model.role == "generate"],
        embedding_models=[model for model in models if model.role == "embedding"],
        models=models,
    )


async def _show_models(client: httpx.AsyncClient, names: list[str]) -> list[dict]:
    results: list[dict] = []
    for name in names:
        response = await client.post("/api/show", json={"model": name})
        response.raise_for_status()
        results.append(response.json())
    return results


@router.post("/embed")
async def embed_texts(payload: EmbedRequest, embedding: EmbeddingDep) -> EmbedResponse:
    try:
        if payload.as_query:
            vectors = [await embedding.aembed_query(text) for text in payload.texts]
        else:
            vectors = await embedding.aembed_documents(payload.texts)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to embed with {embedding.model}. Is the model pulled?",
        ) from exc
    return EmbedResponse(
        model=embedding.model,
        dimensions=len(vectors[0]) if vectors else embedding.dimensions,
        count=len(vectors),
        embeddings=vectors,
    )


@router.post("/references", status_code=status.HTTP_201_CREATED)
async def add_references(payload: ReferenceIn, embedding: EmbeddingDep) -> ReferenceOut:
    metadatas = [{"filename": payload.filename or "inline", "chunk_index": i} for i in range(len(payload.texts))]
    try:
        ids = await reference_index.add_texts(payload.texts, metadatas=metadatas)
    except Exception as exc:
        logger.error("Reference indexing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to index references with {embedding.model}.",
        ) from exc
    return ReferenceOut(ids=ids, count=len(ids), embedding_model=embedding.model)


@router.post("/references/upload", status_code=status.HTTP_201_CREATED)
async def upload_reference(
    embedding: EmbeddingDep,
    file: Annotated[UploadFile, File()],
) -> ReferenceOut:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only UTF-8 text reference files are supported in this version.",
        ) from exc
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reference file is empty.",
        )
    filename = file.filename or "upload.txt"
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
    try:
        ids = await reference_index.add_texts(chunks, metadatas=metadatas)
    except Exception as exc:
        logger.error("Reference upload failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to index references with {embedding.model}.",
        ) from exc
    return ReferenceOut(ids=ids, count=len(ids), embedding_model=embedding.model)


@router.post("/run/upload")
async def run_agent_with_target_file(
    settings: SettingsDep,
    instruction: Annotated[str, Form()],
    generate_model: Annotated[str | None, Form()] = None,
    match_count: Annotated[int, Form()] = 5,
    target: UploadFile | None = None,
) -> RunResponse:
    target_text = None
    if target is not None:
        raw = await target.read()
        try:
            target_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only UTF-8 target files are supported in this version.",
            ) from exc
    if not instruction.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="instruction is required")
    payload = RunRequest(
        instruction=instruction,
        generate_model=generate_model,
        target_text=target_text,
        match_count=match_count,
    )
    return await run_agent(payload, settings)


@router.post("/run")
async def run_agent(payload: RunRequest, settings: SettingsDep) -> RunResponse:
    try:
        result = await run_local_agent(
            instruction=payload.instruction,
            generate_model=payload.generate_model or settings.ollama_model,
            target_text=payload.target_text,
            match_count=payload.match_count,
        )
    except Exception as exc:
        logger.error("Local agent run failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local agent failed. Check that Ollama is running and the generate model is pulled.",
        ) from exc
    return RunResponse.model_validate(result)


__all__ = ["router"]
