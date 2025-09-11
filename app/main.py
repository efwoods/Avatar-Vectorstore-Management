"""
FastAPI ChromaDB Vectorstore Service with S3 Persistence
Main application entry point
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.core.config import get_settings
from app.db.chroma_manager import ChromaDBManager
from app.api.collections import router as collections_router
from app.api.documents import router as documents_router
from app.api.persistence import router as persistence_router
from app.api.health import router as health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global ChromaDB manager instance
chroma_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global chroma_manager
    settings = get_settings()
    
    # Initialize ChromaDB manager
    chroma_manager = ChromaDBManager(settings)
    await chroma_manager.initialize()
    logger.info("ChromaDB manager initialized")
    
    yield
    
    # Cleanup
    if chroma_manager:
        await chroma_manager.cleanup()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="ChromaDB Vectorstore Service",
    description="FastAPI service for managing ChromaDB vectorstore with S3 persistence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get ChromaDB manager
async def get_chroma_manager() -> ChromaDBManager:
    """Get the global ChromaDB manager instance"""
    if chroma_manager is None:
        raise HTTPException(status_code=503, detail="ChromaDB manager not initialized")
    return chroma_manager

# Include routers
app.include_router(
    health_router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    collections_router,
    prefix="/collections",
    tags=["Collections"],
    dependencies=[Depends(get_chroma_manager)]
)

app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(get_chroma_manager)]
)

app.include_router(
    persistence_router,
    prefix="/persistence",
    tags=["Persistence"],
    dependencies=[Depends(get_chroma_manager)]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ChromaDB Vectorstore Service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )