# main.py

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from contextlib import asynccontextmanager
import uvicorn
import logging
import os
import boto3
from chromadb.utils import embedding_functions

from core.config import get_settings
from db.ChromaDBManager import ChromaDBManager
from api.collections import router as collections_router
from api.documents import router as documents_router
from api.persistence import router as persistence_router
from api.health import router as health_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup - Initializing ChromaDB")
    user_id = os.getenv("USER_ID")
    if not user_id:
        logger.error("USER_ID environment variable required")
        raise RuntimeError("USER_ID environment variable required")  # Use RuntimeError for startup
    settings = get_settings()
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region
    )
    
    # Initialize embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    
    # Initialize ChromaDBManager with external clients
    app.state.chroma_manager = ChromaDBManager(
        settings=settings,
        user_id=user_id,
        s3_client=s3_client,
        embedding_function=embedding_function
    )
    await app.state.chroma_manager.initialize()  # Initializes ChromaDB, creates vectorstore if none in S3
    logger.info("ChromaDB initialized successfully")
    yield
    # Shutdown
    logger.info("Application shutdown - Cleaning up ChromaDB")
    if hasattr(app.state, 'chroma_manager'):
        await app.state.chroma_manager.cleanup()
    logger.info("ChromaDB cleanup completed")

app = FastAPI(
    title="ChromaDB Vectorstore Service",
    description="FastAPI service for managing ChromaDB vectorstore with S3 persistence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_chroma_manager():
    if not hasattr(app.state, 'chroma_manager'):
        raise HTTPException(500, "ChromaDB not initialized")
    return app.state.chroma_manager


app.include_router(health_router, prefix="/health", tags=["Health"], dependencies=[Depends(get_chroma_manager)])
app.include_router(collections_router, prefix="/collections", tags=["Collections"], dependencies=[Depends(get_chroma_manager)])
app.include_router(documents_router, prefix="/documents", tags=["Documents"], dependencies=[Depends(get_chroma_manager)])
app.include_router(persistence_router, prefix="/persistence", tags=["Persistence"], dependencies=[Depends(get_chroma_manager)])
    
@app.get("/", tags=["📖 Documentation"])
async def root(request: Request):
    return RedirectResponse(url=f"{request.scope.get('root_path', '')}/docs")