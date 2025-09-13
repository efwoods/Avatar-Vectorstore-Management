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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to hold the chroma manager
chroma_manager_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_manager_instance
    
    # Startup
    logger.info("Application startup - Initializing ChromaDB")
    
    # Get settings and validate required fields
    settings = get_settings()
    user_id = settings.USER_ID
    
    if not user_id:
        logger.error("USER_ID setting is required")
        raise RuntimeError("USER_ID setting is required")
    
    logger.info(f"Starting ChromaDB for user: {user_id}")
    logger.info(f"App: {settings.app_name} v{settings.version}")
    
    # Validate required settings
    if not settings.s3_bucket_name:
        logger.error("S3_BUCKET_NAME is required")
        raise RuntimeError("S3_BUCKET_NAME environment variable is required")
    
    # Initialize S3 client with error handling
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        # Test S3 connection
        s3_client.head_bucket(Bucket=settings.s3_bucket_name)
        logger.info(f"S3 connection successful to bucket: {settings.s3_bucket_name}")
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise RuntimeError(f"S3 initialization failed: {e}")
    
    # Initialize embedding function
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        logger.info(f"Embedding function initialized with model: {settings.embedding_model}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}")
        raise RuntimeError(f"Embedding function initialization failed: {e}")
    
    # Initialize ChromaDBManager with external clients
    try:
        chroma_manager_instance = ChromaDBManager(
            settings=settings,
            user_id=user_id,
            s3_client=s3_client,
            embedding_function=embedding_function
        )
        
        # This will check S3, download if exists, create new if doesn't, and backup
        await chroma_manager_instance.initialize()
        
        logger.info(f"ChromaDB initialized successfully for user: {user_id}")
        logger.info(f"Vectorstore location: {settings.chroma_persist_directory}")
        
        # List existing collections for verification
        collections = await chroma_manager_instance.list_collections()
        logger.info(f"Found {len(collections)} existing collections: {[c['name'] for c in collections]}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB manager: {e}")
        raise RuntimeError(f"ChromaDB initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Application shutdown - Cleaning up ChromaDB")
    try:
        if chroma_manager_instance:
            # Perform final backup before shutdown
            await chroma_manager_instance.backup_to_s3(user_id)
            logger.info("Final backup completed")
            await chroma_manager_instance.cleanup()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
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
    """Dependency function that returns the global chroma manager instance"""
    if chroma_manager_instance is None:
        raise HTTPException(500, "ChromaDB not initialized")
    if chroma_manager_instance.client is None:
        raise HTTPException(500, "ChromaDB client not initialized")
    return chroma_manager_instance

# Include routers WITHOUT dependencies (we'll use Depends() in individual endpoints)
app.include_router(collections_router, prefix="/collections", tags=["Collections"])
app.include_router(documents_router, prefix="/documents", tags=["Documents"]) 
app.include_router(persistence_router, prefix="/persistence", tags=["Persistence"])

@app.get("/", tags=["📖 Documentation"])
async def root(request: Request):
    return RedirectResponse(url=f"{request.scope.get('root_path', '')}/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        manager = get_chroma_manager()
        collections = await manager.list_collections()
        return {
            "status": "healthy",
            "chroma_initialized": manager.client is not None,
            "collections_count": len(collections),
            "user_id": manager.user_id
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }