"""
ChromaDB Manager for handling vectorstore operations and S3 persistence
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import boto3
import json
import os
import shutil
import tempfile
import zipfile
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from core.config import Settings
from db.schema.models import (
    Collection, Document, QueryRequest, QueryResponse,
    BackupRequest, RestoreRequest
)

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages ChromaDB instance with S3 persistence capabilities"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.s3_client = None
        self.embedding_function = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize ChromaDB client and S3 connection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.settings.chroma_persist_directory,
                settings=ChromaSettings(
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.embedding_model
            )
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                region_name=self.settings.aws_region
            )
            
            logger.info("ChromaDB manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB manager: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
    
    # Collection Management
    async def create_collection(self, collection: Collection) -> Dict[str, Any]:
        """Create a new collection"""
        try:
            chroma_collection = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.create_collection(
                    name=collection.name,
                    metadata=collection.metadata or {},
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
            
            result = []
            for collection in collections:
                result.append({
                    "name": collection.name,
                    "id": collection.id,
                    "metadata": collection.metadata,
                    "count": collection.count()
                })
            
            return result
            
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
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            raise
    
    # Document Management
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
            
            # Prepare data for batch insertion
            ids = [doc.id for doc in documents]
            texts = [doc.content for doc in documents]
            metadatas = [doc.metadata or {} for doc in documents]
            embeddings = [doc.embedding for doc in documents if doc.embedding]
            
            # Add documents in batches
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
            
            documents = []
            for i in range(len(result['ids'])):
                documents.append({
                    "id": result['ids'][i],
                    "content": result['documents'][i] if result['documents'] else None,
                    "metadata": result['metadatas'][i] if result['metadatas'] else None,
                    "embedding": result['embeddings'][i] if result['embeddings'] else None
                })
            
            return documents
            
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
    
    # S3 Persistence Methods
    async def backup_to_s3(self, user_id: str, avatar_id: Optional[str] = None) -> Dict[str, Any]:
        """Backup ChromaDB to S3"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Create temporary directory for backup
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_path = os.path.join(temp_dir, "chroma_backup.zip")
                
                # Create zip archive of ChromaDB directory
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: shutil.make_archive(
                        backup_path.replace('.zip', ''),
                        'zip',
                        self.settings.chroma_persist_directory
                    )
                )
                
                # Determine S3 path based on whether avatar_id is provided
                if avatar_id:
                    s3_path = self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
                else:
                    s3_path = self.settings.get_s3_vectorstore_path(user_id)
                
                s3_key = f"{s3_path}backup_{timestamp}.zip"
                
                # Upload to S3
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.upload_file(
                        backup_path,
                        self.settings.s3_bucket_name,
                        s3_key
                    )
                )
                
                # Create metadata
                metadata = {
                    "backup_timestamp": timestamp,
                    "user_id": user_id,
                    "avatar_id": avatar_id,
                    "s3_key": s3_key,
                    "collections": await self.list_collections()
                }
                
                # Upload metadata
                metadata_key = f"{s3_path}metadata_{timestamp}.json"
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.put_object(
                        Bucket=self.settings.s3_bucket_name,
                        Key=metadata_key,
                        Body=json.dumps(metadata),
                        ContentType="application/json"
                    )
                )
                
                logger.info(f"Backup completed for user {user_id}, avatar {avatar_id}")
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to backup to S3: {e}")
            raise
    
    async def restore_from_s3(self, user_id: str, backup_timestamp: str, 
                            avatar_id: Optional[str] = None) -> bool:
        """Restore ChromaDB from S3"""
        try:
            # Determine S3 path
            if avatar_id:
                s3_path = self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
            else:
                s3_path = self.settings.get_s3_vectorstore_path(user_id)
            
            s3_key = f"{s3_path}backup_{backup_timestamp}.zip"
            
            # Create temporary directory for restore
            with tempfile.TemporaryDirectory() as temp_dir:
                backup_path = os.path.join(temp_dir, "chroma_backup.zip")
                
                # Download from S3
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: self.s3_client.download_file(
                        self.settings.s3_bucket_name,
                        s3_key,
                        backup_path
                    )
                )
                
                # Clear current data
                if os.path.exists(self.settings.chroma_persist_directory):
                    shutil.rmtree(self.settings.chroma_persist_directory)
                
                # Extract backup
                with zipfile.ZipFile(backup_path, 'r') as zip_ref:
                    zip_ref.extractall(self.settings.chroma_persist_directory)
                
                # Reinitialize client
                await self.initialize()
                
                logger.info(f"Restore completed for user {user_id}, avatar {avatar_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to restore from S3: {e}")
            raise
    
    async def list_s3_backups(self, user_id: str, avatar_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available S3 backups"""
        try:
            if avatar_id:
                s3_path = self.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
            else:
                s3_path = self.settings.get_s3_vectorstore_path(user_id)
            
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
                        # Get metadata
                        metadata_response = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            lambda: self.s3_client.get_object(
                                Bucket=self.settings.s3_bucket_name,
                                Key=obj['Key']
                            )
                        )
                        
                        metadata = json.loads(metadata_response['Body'].read().decode('utf-8'))
                        backups.append(metadata)
            
            return sorted(backups, key=lambda x: x['backup_timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list S3 backups: {e}")
            raise