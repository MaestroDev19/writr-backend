from fastapi import APIRouter

router = APIRouter(prefix="/cloud", tags=["cloud"])

__all__ = ["router"]
