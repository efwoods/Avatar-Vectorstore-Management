# api/health.py

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime
import logging
import os

from core.config import settings
from db.ChromaDBManager import ChromaDBManager
from db.schema.models import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_chroma_manager() -> ChromaDBManager:
    """Dependency injection placeholder - will be overridden in main.py"""
    pass


@router.get("/",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service, its dependencies, and current user ID"
)
async def health_check(chroma_manager: ChromaDBManager = Depends(get_chroma_manager)):
    """Health check endpoint including USER_ID"""
    user_id = settings.USER_ID
    if not user_id:
        raise HTTPException(400, "USER_ID environment variable not set")

    try:
        # Basic health check
        chroma_status = "healthy"
        s3_status = "healthy"
        
        # Test ChromaDB by listing collections
        try:
            collections = await chroma_manager.list_collections()
            logger.info(f"ChromaDB healthy for user {user_id}: {len(collections)} collections")
        except Exception as e:
            logger.error(f"ChromaDB health check failed for user {user_id}: {e}")
            chroma_status = "unhealthy"

        # Test S3 connectivity by listing backups
        try:
            backups = await chroma_manager.list_s3_backups(user_id)
            logger.info(f"S3 healthy for user {user_id}: {len(backups)} backups")
        except Exception as e:
            logger.error(f"S3 health check failed for user {user_id}: {e}")
            s3_status = "unhealthy"

        overall_status = "healthy" if chroma_status == "healthy" and s3_status == "healthy" else "unhealthy"

        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version=settings.version,
            chroma_status=chroma_status,
            s3_status=s3_status,
            user_id=user_id  # Include USER_ID in response
        )
    except Exception as e:
        logger.error(f"Health check failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@router.get("/ready",
    summary="Readiness check",
    description="Check if the service is ready to accept requests"
)
async def readiness_check(chroma_manager: ChromaDBManager = Depends(get_chroma_manager)):
    """Readiness check endpoint"""
    try:
        # Check ChromaDB readiness
        await chroma_manager.list_collections()
        return {"status": "ready", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

@router.get("/live",
    summary="Liveness check",
    description="Check if the service is alive"
)
async def liveness_check():
    """Liveness check endpoint"""
    return {"status": "alive", "timestamp": datetime.now()}