from fastapi import APIRouter
from .local import router as local_router
from .cloud import router as cloud_router

router = APIRouter()

router.include_router(local_router)
router.include_router(cloud_router)

__all__ = ["router"]
