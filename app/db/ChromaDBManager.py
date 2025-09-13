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
        
    # Fixed ChromaDBManager initialize method
    async def initialize(self):
        """Initialize ChromaDB client, check S3 for vectorstore, and create if absent"""
        try:
            local_persist_dir = self.settings.chroma_persist_directory
            os.makedirs(local_persist_dir, exist_ok=True)

            s3_path = self.settings.get_s3_vectorstore_path(self.user_id)
            logger.info(f"Checking S3 path: {s3_path}")

            vectorstore_exists = await self._check_s3_vectorstore(s3_path)

            if vectorstore_exists:
                logger.info(f"Found backup for user {self.user_id}, restoring...")
                try:
                    await self.restore_from_s3(self.user_id)
                    logger.info(f"Successfully restored from S3 backup for user {self.user_id}")
                except Exception as restore_error:
                    logger.warning(f"Failed to restore from S3 backup: {restore_error}. Creating new vectorstore.")
                    vectorstore_exists = False

            if not vectorstore_exists:
                logger.info(f"No existing vectorstore found. Creating new vectorstore for user {self.user_id}")
                await self._create_fresh_vectorstore(local_persist_dir)

                # Create an initial backup after fresh vectorstore creation (if auto-backup enabled)
                if self.settings.auto_backup_enabled:
                    try:
                        await self.backup_to_s3(self.user_id)
                        logger.info(f"Created initial backup for fresh vectorstore for user {self.user_id}")
                    except Exception as backup_error:
                        logger.warning(f"Failed to create initial backup: {backup_error}. Continuing without backup.")

            await self._ensure_client_ready()
            await self._validate_client()

            # Start auto-backup scheduler if enabled
            if self.settings.auto_backup_enabled:
                self._schedule_auto_backup()

            logger.info(f"ChromaDB manager initialized for user {self.user_id}")
            logger.info(f"Vectorstore directory: {local_persist_dir}")
            logger.info(f"S3 backup path: {s3_path}")
            logger.info(f"Auto-backup: {'enabled' if self.settings.auto_backup_enabled else 'disabled'}")

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


    def _schedule_auto_backup(self):
        """Schedule automatic backups if enabled"""
        import asyncio
        from datetime import timedelta

        async def auto_backup_task():
            while True:
                try:
                    # Wait for the backup interval
                    await asyncio.sleep(self.settings.backup_interval_hours * 3600)  # Convert hours to seconds

                    # Perform backup
                    logger.info(f"Performing scheduled backup for user {self.user_id}")
                    await self.backup_to_s3(self.user_id)
                    logger.info(f"Scheduled backup completed for user {self.user_id}")

                except asyncio.CancelledError:
                    logger.info(f"Auto-backup task cancelled for user {self.user_id}")
                    break
                except Exception as e:
                    logger.error(f"Auto-backup failed for user {self.user_id}: {e}")
                    # Continue the loop, don't break on individual backup failures

        # Store the task reference for cleanup
        if not hasattr(self, '_backup_task'):
            self._backup_task = asyncio.create_task(auto_backup_task())
            logger.info(f"Auto-backup scheduled every {self.settings.backup_interval_hours} hours")

    async def cleanup(self):
        """Clean up resources including auto-backup task"""
        # Cancel auto-backup task if running
        if hasattr(self, '_backup_task') and not self._backup_task.done():
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            logger.info("Auto-backup task cancelled")

        if self.executor:
            self.executor.shutdown(wait=True)
        self.client = None

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

    # Fixed backup and restore methods for ChromaDBManager
    async def backup_to_s3(self, user_id: str) -> Dict[str, Any]:
        """Backup ChromaDB to S3 - single file, no timestamp folders"""
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized")
        try:
            timestamp = datetime.now().isoformat()
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_zip = os.path.join(temp_dir, "backup.zip")
                shutil.make_archive(
                    backup_zip.replace(".zip", ""), "zip", self.settings.chroma_persist_directory
                )

                # Store directly in vectorstore path, no timestamp subfolder
                s3_path = self.settings.get_s3_vectorstore_path(user_id)
                backup_key = f"{s3_path}chroma_backup.zip"
                metadata_key = f"{s3_path}metadata.json"

                # Upload backup file
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.upload_file(
                        backup_zip, self.settings.s3_bucket_name, backup_key
                    )
                )

                # Upload metadata
                metadata = {
                    "user_id": user_id, 
                    "backup_timestamp": timestamp,
                    "backup_key": backup_key
                }
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.put_object(
                        Bucket=self.settings.s3_bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata)
                    )
                )

                logger.info(f"Backup completed: {backup_key}")
                return metadata
        except Exception as e:
            logger.error(f"Backup to S3 failed: {e}")
            raise

    async def restore_from_s3(self, user_id: str, timestamp: str = None):
        """Restore ChromaDB from S3 - use single backup file"""
        if self.s3_client is None:
            raise RuntimeError("S3 client not initialized")
        try:
            s3_path = self.settings.get_s3_vectorstore_path(user_id)
            backup_key = f"{s3_path}chroma_backup.zip"

            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "restore.zip")
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.download_file(
                        self.settings.s3_bucket_name, backup_key, zip_path
                    )
                )

                # Clean existing directory before restore
                if os.path.exists(self.settings.chroma_persist_directory):
                    await self._safe_rmtree(self.settings.chroma_persist_directory)

                os.makedirs(self.settings.chroma_persist_directory, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.settings.chroma_persist_directory)

            logger.info(f"Restore completed from: {backup_key}")
        except Exception as e:
            logger.error(f"Restore from S3 failed: {e}")
            raise

    async def _check_s3_vectorstore(self, s3_path: str) -> bool:
        """Check if vectorstore backup exists in S3"""
        if self.s3_client is None:
            return False
        try:
            backup_key = f"{s3_path}chroma_backup.zip"
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.s3_client.head_object(
                    Bucket=self.settings.s3_bucket_name, Key=backup_key
                )
            )
            logger.info(f"Found existing backup: {backup_key}")
            return True
        except self.s3_client.exceptions.NoSuchKey:
            logger.info(f"No existing backup found at: {backup_key}")
            return False
        except Exception as e:
            logger.error(f"Failed to check S3 vectorstore: {e}")
            return False

    async def list_s3_backups(self, user_id: str) -> List[Dict[str, Any]]:
        """List available backups for user"""
        if self.s3_client is None:
            return []
        try:
            s3_path = self.settings.get_s3_vectorstore_path(user_id)
            metadata_key = f"{s3_path}metadata.json"

            try:
                meta_resp = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.get_object(
                        Bucket=self.settings.s3_bucket_name, Key=metadata_key
                    )
                )
                metadata = json.loads(meta_resp['Body'].read().decode('utf-8'))
                return [metadata]
            except self.s3_client.exceptions.NoSuchKey:
                return []
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    async def _get_latest_backup(self, s3_path: str) -> Optional[Dict[str, Any]]:
        """Get latest backup metadata"""
        backups = await self.list_s3_backups(self.user_id)
        return backups[0] if backups else None

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

# Add this method to your ChromaDBManager class in ChromaDBManager.py

    async def _ensure_client_ready(self):
        """Ensure ChromaDB client is properly initialized"""
        if self.client is None:
            try:
                logger.info("Initializing ChromaDB client...")
                self.client = chromadb.PersistentClient(
                    path=self.settings.chroma_persist_directory,
                    settings=ChromaSettings(
                        allow_reset=True,
                        is_persistent=True,
                        anonymized_telemetry=False
                    )
                )
                
                # Test the client connection
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    lambda: self.client.list_collections()
                )
                
                logger.info(f"ChromaDB client initialized successfully at {self.settings.chroma_persist_directory}")
                
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                self.client = None  # Reset to None on failure
                raise RuntimeError(f"ChromaDB client initialization failed: {e}")
        
        # Additional safety check - make sure client is still responsive
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                lambda: self.client.list_collections()
            )
        except Exception as e:
            logger.warning(f"ChromaDB client not responsive, reinitializing: {e}")
            self.client = None
            await self._ensure_client_ready()  # Recursive call to reinitialize