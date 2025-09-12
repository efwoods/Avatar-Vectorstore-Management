# db/ChromaDBManager.py

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
            # Set local persist directory
            local_persist_dir = self.settings.chroma_persist_directory

            # Check if vectorstore exists in S3
            s3_path = self.settings.get_s3_vectorstore_path(self.user_id)
            vectorstore_exists = await self._check_s3_vectorstore(s3_path)

            if vectorstore_exists:
                # Try to restore from S3 first
                latest_backup = await self._get_latest_backup(s3_path)
                if latest_backup:
                    logger.info(f"Found backup for user {self.user_id}, restoring from {latest_backup['backup_timestamp']}")
                    try:
                        await self.restore_from_s3(self.user_id, latest_backup['backup_timestamp'])
                        logger.info(f"Successfully restored from S3 backup for user {self.user_id}")
                        return
                    except Exception as restore_error:
                        logger.warning(f"Failed to restore from S3 backup: {restore_error}. Creating new vectorstore.")
                        vectorstore_exists = False  # Fall through to create new

            # Create new vectorstore (either no S3 backup exists or restore failed)
            if not vectorstore_exists:
                logger.info(f"Creating new vectorstore for user {self.user_id}")
                await self._create_fresh_vectorstore(local_persist_dir)

            logger.info(f"ChromaDB manager initialized for user {self.user_id}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager for user {self.user_id}: {e}")
            raise

    async def _check_s3_vectorstore(self, s3_path: str) -> bool:
        """Check if vectorstore backups exist in S3"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name,
                    Prefix=s3_path
                )
            )
            # Check if there are any objects in the S3 path
            return 'Contents' in response and len(response['Contents']) > 0
        except Exception as e:
            logger.error(f"Failed to check S3 vectorstore: {e}")
            return False

    async def _get_latest_backup(self, s3_path: str) -> Optional[Dict[str, Any]]:
        """Get the latest backup metadata from S3"""
        backups = await self.list_s3_backups(self.user_id)
        return backups[0] if backups else None
    
    async def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    async def create_collection(self, collection: Collection) -> Dict[str, Any]:
        """Create a new collection"""
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
    
    async def backup_to_s3(self, user_id: str, avatar_id: Optional[str] = None) -> Dict[str, Any]:
        """Backup ChromaDB to S3"""
        try:
            timestamp = datetime.now().isoformat()
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_path = os.path.join(temp_dir, "chroma_backup.zip")
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: shutil.make_archive(
                        backup_path.replace('.zip', ''),
                        'zip',
                        self.settings.chroma_persist_directory
                    )
                )
                s3_path = self.settings.get_s3_vectorstore_path(user_id) if not avatar_id else self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
                s3_key = f"{s3_path}backup_{timestamp}.zip"
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.upload_file(
                        backup_path,
                        self.settings.s3_bucket_name,
                        s3_key
                    )
                )
                metadata = {
                    "backup_timestamp": timestamp,
                    "user_id": user_id,
                    "avatar_id": avatar_id,
                    "s3_key": s3_key,
                    "collections": await self.list_collections()
                }
                metadata_key = f"{s3_path}metadata_{timestamp}.json"
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.put_object(
                        Bucket=self.settings.s3_bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata, default=str),
                        ContentType="application/json"
                    )
                )
                logger.info(f"Backup completed for user {user_id}, avatar {avatar_id}")
                return metadata
        except Exception as e:
            logger.error(f"Failed to backup to S3: {e}")
            raise
    

    async def restore_from_s3(self, user_id: str, backup_timestamp: str, avatar_id: Optional[str] = None) -> bool:
        """Restore ChromaDB from S3"""
        try:
            s3_path = self.settings.get_s3_vectorstore_path(user_id) if not avatar_id else self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
            s3_key = f"{s3_path}backup_{backup_timestamp}.zip"
            local_persist_dir = self.settings.chroma_persist_directory

            with tempfile.TemporaryDirectory() as temp_dir:
                backup_path = os.path.join(temp_dir, "chroma_backup.zip")

                # Download backup from S3
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.download_file(
                        self.settings.s3_bucket_name,
                        s3_key,
                        backup_path
                    )
                )

                # Clean and recreate local directory with retry logic
                if os.path.exists(local_persist_dir):
                    await self._safe_rmtree(local_persist_dir)
                os.makedirs(local_persist_dir, exist_ok=True)

                # Extract backup
                with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                    zip_ref.extractall(local_persist_dir)

                # Initialize ChromaDB client with restored data
                self.client = chromadb.PersistentClient(
                    path=local_persist_dir,
                    settings=ChromaSettings(
                        allow_reset=True,
                        is_persistent=True,
                        anonymized_telemetry=False
                    )
                )

                # Verify the restored database works
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.client.list_collections()
                )

                logger.info(f"Restore completed for user {user_id}, avatar {avatar_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to restore from S3: {e}")
            raise

    async def list_s3_backups(self, user_id: str, avatar_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available S3 backups"""
        try:
            s3_path = self.settings.get_s3_vectorstore_path(user_id) if not avatar_id else self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
            response = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.settings.s3_bucket_name,
                    Prefix=s3_path
                )
            )
            backups = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('metadata.json'):
                        try:
                            metadata_response = await asyncio.get_event_loop().run_in_executor(
                                self.executor,
                                lambda: self.s3_client.get_object(
                                    Bucket=self.settings.s3_bucket_name,
                                    Key=obj['Key']
                                )
                            )
                            metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
                            backups.append(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to read metadata from {obj['Key']}: {e}")
                            continue
            return sorted(backups, key=lambda x: x.get('backup_timestamp', ''), reverse=True)
        except Exception as e:
            logger.error(f"Failed to list S3 backups: {e}")
            return []
        
    async def _create_fresh_vectorstore(self, local_persist_dir: str):
        """Create a completely fresh ChromaDB vectorstore"""
        try:
            logger.info(f"Starting fresh vectorstore creation at {local_persist_dir}")

            # Clean directory with retry logic
            # if os.path.exists(local_persist_dir):
                # await self._safe_rmtree(local_persist_dir)
            os.makedirs(local_persist_dir, exist_ok=True)

            # Initialize ChromaDB client with minimal settings
            self.client = chromadb.PersistentClient(
                path=local_persist_dir,
                settings=ChromaSettings(
                    allow_reset=True,
                    is_persistent=True,
                    anonymized_telemetry=False
                )
            )

            logger.info(f"ChromaDB client created: {self.client is not None}")

            # Wait a moment for initialization
            await asyncio.sleep(0.5)

            # Force database initialization by attempting to list collections
            # This should create the internal SQLite tables if they don't exist
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._safe_list_collections
            )

            # Create default collection with explicit error handling
            await self._create_default_collection_safe()

            # Backup the new vectorstore to S3
            await self.backup_to_s3(self.user_id)

            logger.info(f"Fresh vectorstore creation completed. Client status: {self.client is not None}")

        except Exception as e:
            logger.error(f"Failed to create fresh vectorstore: {e}")
            raise

    def _safe_list_collections(self):
        """Safely list collections with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
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

    async def _safe_rmtree(self, path: str):
        """Safely remove directory tree with retry logic for busy errors"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(path)
                logger.info(f"Successfully removed directory {path}")
                return
            except OSError as e:
                if e.errno == 16:  # Device or resource busy
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                        logger.warning(f"Directory {path} busy on attempt {attempt + 1}, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed to remove directory {path} after {max_retries} attempts")
                        raise
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error removing {path}: {e}")
                raise