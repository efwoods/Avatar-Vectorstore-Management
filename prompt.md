
I need to check if there is a chroma_db in the s3 bucket for the user_id that is in the .env;
If it exists, I need to download it and start it on startup. 
If it doesn't , I need to create a new chroma_db and back it up to the s3_bucket, and use the chroma_db that has been created.
The chromadb needs to be created under 
users/{user_id}/
├── vectorstore/     


## Directory Structure

.
├── .github
│   └── workflows
│       └── deploy.yml # This is used in the github actions to build the docker image and save it in docker hub. This github actions will then use the docker hub image in google cloud run and run the container in google cloud run.
├── .gitignore
├── app
│   ├── api # This holds api routes. Each route is included in main.py
│   │   └── __init__.py
│   ├── classes # These hold class definitions
│   │   └── __init__.py 
│   ├── core # This holds settings and configuration options
│   │   └── __init__.py
│   ├── db # This holds the database instance and class
│   │   ├── __init__.py
│   │   └── schema # this holds pydantic models
│   │       └── __init__.py
│   ├── __init__.py
│   ├── models # This holds huggingface models
│   │   └── __init__.py
│   └── service # This holds utility functions and function definitions used in the other files
│       └── __init__.py
├── docker-compose.yml # This is used to hot-reload and local dev testing (Dockerfile.dev)
├── Dockerfile.dev # This is used for the local dev testing image
├── Dockerfile # This is used for the production image in the google deploy script
├── prompt.md
├── README.md
└── requirements.txt

## S3_persistence Structure:
This is the S3 Persistence Structure
users/{user_id}/
├── vectorstore/                   # User-level vectorstore (chroma_db)
├── avatars/{avatar_id}/
│   ├── vectorstore_data/          # Avatar-specific context data (preprocessed);
|   ├── vectorstore_metadata/ There is a meta datafile dictionary of booleans determining if each vectorstore_data object is used for training
│   ├── adapters/                  # QLoRA adapter files (the actual Adapter is stored here)
│   ├── adapters/training_data/    # Training data for fine-tuning (preprocessed for the LoRA Adapter); 
│   ├── adapters/metadata/    # There is a meta datafile dictionary of booleans determining if each adapters/training_data object is used for training.
|   ├── media/audio/               # Processed audio (audio only of the target avatar speaking)
|   ├── media/original             # Unprocessed, original media for a specific avatar (audio/video/images/documents)
|   ├── media/original/video               # Original video 
|   ├── media/original/text                # Original text documents 
|   ├── media/original/audio               # Original audio  
|   └── media/original/images
|── image/                         # User-level personal image
|── *{other_potential_user_level_folders}  # Other potential user-level folders such as billing & account information


------



---

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import boto3
import json
import os
import shutil
import tempfile
import zipfile
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.config import Settings
from db.schema.models import Collection, Document, QueryRequest, QueryResponse, BackupRequest, RestoreRequest

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages ChromaDB instance with S3 persistence capabilities"""
    
    def __init__(self, settings: Settings, user_id: str, s3_client: boto3.client, embedding_function: Any):
        self.settings = settings
        self.user_id = user_id
        self.client = None
        self.s3_client = s3_client
        self.embedding_function = embedding_function
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def initialize(self):
        """Initialize ChromaDB client, check S3 for vectorstore, and create if absent"""
        try:
            local_persist_dir = self.settings.chroma_persist_directory
            os.makedirs(local_persist_dir, exist_ok=True)
            
            s3_path = self.settings.get_s3_vectorstore_path(self.user_id)
            vectorstore_exists = await self._check_s3_vectorstore(s3_path)
            
            if vectorstore_exists:
                latest_backup = await self._get_latest_backup(s3_path)
                if latest_backup:
                    logger.info(f"Found backup for user {self.user_id}, restoring from {latest_backup['backup_timestamp']}")
                    try:
                        await self.restore_from_s3(self.user_id, latest_backup['backup_timestamp'])
                        logger.info(f"Successfully restored from S3 backup for user {self.user_id}")
                    except Exception as restore_error:
                        logger.warning(f"Failed to restore from S3 backup: {restore_error}. Creating new vectorstore.")
                        vectorstore_exists = False
            
            if not vectorstore_exists:
                logger.info(f"Creating new vectorstore for user {self.user_id}")
                await self._create_fresh_vectorstore(local_persist_dir)
                # Create an initial backup after fresh vectorstore creation
                try:
                    await self.backup_to_s3(self.user_id)
                    logger.info(f"Created initial backup for fresh vectorstore for user {self.user_id}")
                except Exception as backup_error:
                    logger.warning(f"Failed to create initial backup: {backup_error}. Continuing without backup.")
            
            await self._ensure_client_ready()
            await self._validate_client()
            
            logger.info(f"ChromaDB manager initialized for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager for user {self.user_id}: {e}")
            raise
        
    async def _validate_client(self):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, lambda: self.client.list_collections()
                )
                logger.info("ChromaDB client validated successfully")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Client validation failed (attempt {attempt+1}): {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)
                    await self._ensure_client_ready()
                else:
                    raise RuntimeError(f"Failed to validate ChromaDB client after {max_retries} attempts: {e}")

    async def _ensure_client_ready(self):
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=ChromaSettings(
                    allow_reset=True,
                    is_persistent=True,
                    anonymized_telemetry=False
                )
            )

    async def _create_fresh_vectorstore(self, persist_dir: str):
        """Create a fresh vectorstore, safely handling directory cleanup"""
        try:
            # First, ensure any existing client is cleaned up
            if self.client is not None:
                self.client = None
                # Give time for any file handles to be released
                await asyncio.sleep(1)

            # Check if directory exists and has contents
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                logger.info(f"Directory {persist_dir} exists with contents, attempting cleanup...")
                await self._safe_rmtree(persist_dir)

            # Ensure directory exists
            os.makedirs(persist_dir, exist_ok=True)

            # Create new client
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    allow_reset=True,
                    is_persistent=True,
                    anonymized_telemetry=False
                )
            )

            # Verify the client is working by doing a simple operation
            try:
                collections = self.client.list_collections()
                logger.info(f"Fresh vectorstore created successfully with {len(collections)} collections")
            except Exception as verify_error:
                logger.error(f"Failed to verify fresh vectorstore: {verify_error}")
                raise
            
            logger.info(f"Successfully created fresh vectorstore at {persist_dir}")

        except Exception as e:
            logger.error(f"Failed to create fresh vectorstore: {e}")
            # If we can't clean up, try to work with existing directory
            if os.path.exists(persist_dir):
                logger.warning("Attempting to use existing directory without cleanup")
                try:
                    self.client = chromadb.PersistentClient(
                        path=persist_dir,
                        settings=ChromaSettings(
                            allow_reset=True,
                            is_persistent=True,
                            anonymized_telemetry=False
                        )
                    )
                    # Test the fallback client
                    self.client.list_collections()
                    logger.info("Successfully initialized with existing directory")
                except Exception as fallback_error:
                    logger.error(f"Fallback client initialization also failed: {fallback_error}")
                    raise
            else:
                raise
            
    async def _safe_rmtree(self, path: str):
        """Safely remove directory tree with retry logic for busy errors"""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                # Try to remove individual files first
                if os.path.isdir(path):
                    for root, dirs, files in os.walk(path, topdown=False):
                        # Remove files
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                os.chmod(file_path, 0o777)  # Ensure writable
                                os.remove(file_path)
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Failed to remove file {file_path}: {e}")
                        
                        # Remove empty directories
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            try:
                                os.rmdir(dir_path)
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Failed to remove directory {dir_path}: {e}")
                    
                    # Finally remove the root directory
                    os.rmdir(path)
                
                logger.info(f"Successfully removed directory {path}")
                return
                
            except OSError as e:
                if e.errno == 16:  # Device or resource busy
                    if attempt < max_retries - 1:
                        wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
                        logger.warning(f"Directory {path} busy on attempt {attempt + 1}, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed to remove directory {path} after {max_retries} attempts, continuing anyway...")
                        # Don't raise - let the application continue with existing directory
                        return
                else:
                    logger.error(f"Failed to remove directory {path}: {e}")
                    # Don't raise for other OS errors either
                    return
            except Exception as e:
                logger.error(f"Unexpected error removing {path}: {e}")
                return

    async def _check_s3_vectorstore(self, s3_path: str) -> bool:
        if self.s3_client is None:
            return False
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name, Prefix=s3_path
                )
            )
            return 'Contents' in response and len(response['Contents']) > 0
        except Exception as e:
            logger.error(f"Failed to check S3 vectorstore: {e}")
            return False

    async def list_s3_backups(self, user_id: str) -> List[Dict[str, Any]]:
        if self.s3_client is None:
            return []
        try:
            s3_path = self.settings.get_s3_vectorstore_path(user_id)
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name, Prefix=s3_path
                )
            )
            backups = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('metadata.json'):
                        meta_resp = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: self.s3_client.get_object(
                                Bucket=self.settings.s3_bucket_name, Key=obj['Key']
                            )
                        )
                        metadata = json.loads(meta_resp['Body'].read().decode('utf-8'))
                        backups.append(metadata)
            return sorted(backups, key=lambda x: x.get('backup_timestamp', ''), reverse=True)
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    async def _get_latest_backup(self, s3_path: str) -> Optional[Dict[str, Any]]:
        backups = await self.list_s3_backups(self.user_id)
        return backups[0] if backups else None

    async def backup_to_s3(self, user_id: str) -> Dict[str, Any]:
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized")
        try:
            timestamp = datetime.now().isoformat()
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_zip = os.path.join(temp_dir, "backup.zip")
                shutil.make_archive(
                    backup_zip.replace(".zip", ""), "zip", self.settings.chroma_persist_directory
                )
                
                key = f"{self.settings.get_s3_vectorstore_path(user_id)}/{timestamp}/chroma_backup.zip"
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.upload_file(
                        backup_zip, self.settings.s3_bucket_name, key
                    )
                )
                
                metadata = {"user_id": user_id, "backup_timestamp": timestamp}
                meta_key = f"{self.settings.get_s3_vectorstore_path(user_id)}/{timestamp}/metadata.json"
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.put_object(
                        Bucket=self.settings.s3_bucket_name,
                        Key=meta_key,
                        Body=json.dumps(metadata)
                    )
                )
                
                return metadata
        except Exception as e:
            logger.error(f"Backup to S3 failed: {e}")
            raise

    async def restore_from_s3(self, user_id: str, timestamp: str):
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized")
        try:
            key = f"{self.settings.get_s3_vectorstore_path(user_id)}/{timestamp}/chroma_backup.zip"
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "restore.zip")
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.download_file(
                        self.settings.s3_bucket_name, key, zip_path
                    )
                )
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.settings.chroma_persist_directory)
        except Exception as e:
            logger.error(f"Restore from S3 failed: {e}")
            raise

    async def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.client = None  # Explicitly set to None to avoid stale references

    async def create_collection(self, collection: Collection) -> Dict[str, Any]:
        """Create a new collection"""
        await self._ensure_client_ready()
        metadata = collection.metadata or {}
        if not metadata:
            metadata = {"created_by": self.user_id, "created_at": datetime.now().isoformat()}
        
        try:
            chroma_collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.create_collection(
                    name=collection.name,
                    metadata=metadata,
                    embedding_function=self.embedding_function
                )
            )
            # Backup to S3 after creating collection
            await self.backup_to_s3(self.user_id)
            return {
                "name": chroma_collection.name,
                "id": chroma_collection.id,
                "metadata": chroma_collection.metadata,
                "count": chroma_collection.count()
            }
        except Exception as e:
            logger.error(f"Failed to create collection {collection.name}: {e}")
            raise

    async def get_collection(self, name: str) -> Dict[str, Any]:
        """Get collection by name"""
        await self._ensure_client_ready()
        try:
            chroma_collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=name,
                    embedding_function=self.embedding_function
                )
            )
            return {
                "name": chroma_collection.name,
                "id": chroma_collection.id,
                "metadata": chroma_collection.metadata,
                "count": chroma_collection.count()
            }
        except Exception as e:
            logger.error(f"Failed to get collection {name}: {e}")
            raise

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections"""
        await self._ensure_client_ready()
        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.list_collections()
            )
            return [
                {
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata,
                    "count": collection.count()
                }
                for collection in collections
            ]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        await self._ensure_client_ready()
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.delete_collection(name=name)
            )
            # Backup to S3 after deletion
            await self.backup_to_s3(self.user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            raise

    async def add_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Add documents to a collection"""
        await self._ensure_client_ready()
        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
            
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            
            batch_size = self.settings.max_batch_size
            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size] if embeddings else None
                
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        embeddings=batch_embeddings
                    )
                )
            
            # Backup to S3 after adding documents
            await self.backup_to_s3(self.user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_name}: {e}")
            raise

    async def get_documents(self, collection_name: str, ids: Optional[List[str]] = None, 
                          limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents from a collection"""
        await self._ensure_client_ready()
        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
            
            if ids:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: collection.get(ids=ids)
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: collection.get(limit=limit, offset=offset)
                )
            
            return [
                {
                    "id": result['ids'][i],
                    "content": result['documents'][i] if result['documents'] else None,
                    "metadata": result['metadatas'][i] if result['metadatas'] else None,
                    "embedding": result['embeddings'][i] if result['embeddings'] else None
                }
                for i in range(len(result['ids']))
            ]
        except Exception as e:
            logger.error(f"Failed to get documents from collection {collection_name}: {e}")
            raise

    async def update_documents(self, collection_name: str, documents: List[Document]) -> bool:
        """Update documents in a collection"""
        await self._ensure_client_ready()
        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
            
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: collection.update(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings if embeddings else None
                )
            )
            # Backup to S3 after updating documents
            await self.backup_to_s3(self.user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update documents in collection {collection_name}: {e}")
            raise

    async def delete_documents(self, collection_name: str, ids: List[str]) -> bool:
        """Delete documents from a collection"""
        await self._ensure_client_ready()
        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: collection.delete(ids=ids)
            )
            # Backup to S3 after deletion
            await self.backup_to_s3(self.user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from collection {collection_name}: {e}")
            raise

    async def query_collection(self, collection_name: str, query: QueryRequest) -> QueryResponse:
        """Query a collection"""
        await self._ensure_client_ready()
        try:
            collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
            )
            
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: collection.query(
                    query_texts=query.query_texts,
                    query_embeddings=query.query_embeddings,
                    n_results=query.n_results,
                    where=query.where,
                    where_document=query.where_document,
                    include=query.include or ["metadatas", "documents", "distances"]
                )
            )
            
            return QueryResponse(
                ids=results['ids'],
                embeddings=results.get('embeddings'),
                documents=results.get('documents'),
                metadatas=results.get('metadatas'),
                distances=results.get('distances')
            )
        except Exception as e:
            logger.error(f"Failed to query collection {collection_name}: {e}")
            raise

    def _safe_list_collections(self):
        """Safely list collections with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.client is None:
                    raise RuntimeError("Client not initialized")
                return self.client.list_collections()
            except Exception as e:
                if "no such table" in str(e).lower() and attempt < max_retries - 1:
                    # ChromaDB needs more time to initialize - wait and retry
                    import time
                    time.sleep(1)
                    continue
                else:
                    raise

    async def _create_default_collection_safe(self):
        """Safely create default collection with retry logic"""
        await self._ensure_client_ready()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                metadata = {"created_by": self.user_id, "created_at": datetime.now().isoformat()}
                # Try creating collection directly without async wrapper first
                collection = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.create_collection(
                        name="default",
                        metadata=metadata,
                        embedding_function=self.embedding_function
                    )
                )
                logger.info(f"Successfully created default collection for user {self.user_id}")
                return
            except Exception as e:
                if "no such table" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database not ready on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2)  # Wait longer between retries
                    continue
                else:
                    logger.error(f"Failed to create default collection after {max_retries} attempts: {e}")
                    raise


                    ----
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

