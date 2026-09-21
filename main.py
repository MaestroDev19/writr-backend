from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import router   
from utils import logger

app = FastAPI()

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
