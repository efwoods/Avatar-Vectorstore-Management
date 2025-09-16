"""
Pydantic models for ChromaDB Vectorstore Service
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Base Models
class Collection(BaseModel):
    """Collection model"""
    name: str = Field(..., description="Collection name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")

class Document(BaseModel):
    """Document model"""
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding")

class QueryRequest(BaseModel):
    """Query request model"""
    query_texts: Optional[List[str]] = Field(None, description="Query texts")
    query_embeddings: Optional[List[List[float]]] = Field(None, description="Query embeddings")
    n_results: int = Field(10, description="Number of results to return")
    where: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    where_document: Optional[Dict[str, str]] = Field(None, description="Document content filter")
    include: Optional[List[str]] = Field(None, description="Fields to include in response")

class QueryResponse(BaseModel):
    """Query response model"""
    ids: List[List[str]] = Field(..., description="Document IDs")
    embeddings: Optional[List[List[List[float]]]] = Field(None, description="Document embeddings")
    documents: Optional[List[List[str]]] = Field(None, description="Document contents")
    metadatas: Optional[List[List[Dict[str, Any]]]] = Field(None, description="Document metadata")
    distances: Optional[List[List[float]]] = Field(None, description="Distance scores")

# API Request/Response Models
class CreateCollectionRequest(BaseModel):
    """Create collection request"""
    name: str = Field(..., description="Collection name", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")

class CreateCollectionResponse(BaseModel):
    """Create collection response"""
    name: str
    id: str
    metadata: Dict[str, Any]
    count: int
    message: str = "Collection created successfully"

class CollectionResponse(BaseModel):
    """Collection response"""
    name: str
    id: str
    metadata: Dict[str, Any]
    count: int

class AddDocumentsRequest(BaseModel):
    """Add documents request"""
    documents: List[Document] = Field(..., description="Documents to add")
    collection_name: str = Field(..., description="Collection name")

class AddDocumentsResponse(BaseModel):
    """Add documents response"""
    success: bool
    message: str
    count: int

class GetDocumentsRequest(BaseModel):
    """Get documents request"""
    collection_name: str = Field(..., description="Collection name")
    ids: Optional[List[str]] = Field(None, description="Document IDs to retrieve")
    limit: Optional[int] = Field(None, description="Maximum number of documents to return")
    offset: Optional[int] = Field(None, description="Number of documents to skip")

class GetDocumentsResponse(BaseModel):
    """Get documents response"""
    documents: List[Dict[str, Any]]
    count: int

class UpdateDocumentsRequest(BaseModel):
    """Update documents request"""
    documents: List[Document] = Field(..., description="Documents to update")
    collection_name: str = Field(..., description="Collection name")

class UpdateDocumentsResponse(BaseModel):
    """Update documents response"""
    success: bool
    message: str
    updated_count: int

class DeleteDocumentsRequest(BaseModel):
    """Delete documents request"""
    collection_name: str = Field(..., description="Collection name")
    ids: List[str] = Field(..., description="Document IDs to delete")

class DeleteDocumentsResponse(BaseModel):
    """Delete documents response"""
    success: bool
    message: str
    deleted_count: int

class QueryCollectionRequest(BaseModel):
    """Query collection request"""
    collection_name: str = Field(..., description="Collection name")
    query: QueryRequest = Field(..., description="Query parameters")

# Persistence Models
class BackupRequest(BaseModel):
    """Backup request"""
    user_id: str = Field(..., description="User ID")
    avatar_id: Optional[str] = Field(None, description="Avatar ID (optional)")

class BackupResponse(BaseModel):
    """Backup response"""
    success: bool
    message: str
    backup_info: Dict[str, Any]

class RestoreRequest(BaseModel):
    """Restore request"""
    user_id: str = Field(..., description="User ID")
    backup_timestamp: str = Field(..., description="Backup timestamp")
    avatar_id: Optional[str] = Field(None, description="Avatar ID (optional)")

class RestoreResponse(BaseModel):
    """Restore response"""
    success: bool
    message: str

class ListBackupsRequest(BaseModel):
    """List backups request"""
    user_id: str = Field(..., description="User ID")
    avatar_id: Optional[str] = Field(None, description="Avatar ID (optional)")

class ListBackupsResponse(BaseModel):
    """List backups response"""
    backups: List[Dict[str, Any]]
    count: int

# Health Models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    chroma_status: str
    s3_status: str

# Error Models
class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    message: str
    timestamp: datetime

class SimpleQueryRequest(BaseModel):
    """Simplified query request - just pass the query text"""
    query: str = Field(..., description="Query text to search for")
    n_results: int = Field(10, description="Number of results to return")

class SimpleQueryResponse(BaseModel):
    """Simplified query response"""
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    count: int = Field(..., description="Number of results returned")