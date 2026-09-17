from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.local_db import get_engine
from core.local_embedding import get_local_embedding
from core.local_vector_store import get_local_vector_store
from router import router
from utils import logger


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_engine()
    store = get_local_vector_store()
    logger.info("Reference vector store ready (%s dims)", store.ndim)
    try:
        embedding = get_local_embedding()
        dim = await embedding.awarmup()
        logger.info("Local embedding ready: %s (%s dims)", embedding.model, dim)
    except Exception as exc:
        logger.warning("Ollama embedding warmup skipped: %s", exc)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def read_health() -> dict[str, str]:
    logger.info("GET request received at health endpoint")
    return {"message": "healthy"}


@app.get("/")
def read_root() -> dict[str, str]:
    logger.info("GET request received at root endpoint")
    return {"message": "welcome to Writr"}
