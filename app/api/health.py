"""
Health check API router
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import logging

from core.config import get_settings
from db.schema.models import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the service and its dependencies"
)
async def health_check():
    """Health check endpoint"""
    settings = get_settings()
    
    try:
        # Basic health check
        chroma_status = "healthy"
        s3_status = "healthy"
        
        # You could add more sophisticated checks here
        # For example, test ChromaDB connection, S3 connectivity, etc.
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version=settings.version,
            chroma_status=chroma_status,
            s3_status=s3_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

@router.get("/ready",
    summary="Readiness check",
    description="Check if the service is ready to accept requests"
)
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Add readiness checks here (database connectivity, etc.)
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