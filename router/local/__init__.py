from fastapi import APIRouter

router = APIRouter(prefix="/local", tags=["local"])

__all__ = ["router"]
