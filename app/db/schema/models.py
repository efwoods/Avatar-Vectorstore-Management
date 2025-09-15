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

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryResponse(BaseModel):
    collection_name: str
    query: str
    context: str  # Ready-to-use context string for model prompts
    documents: List[str]  # Individual documents
    raw_results: Dict[str, Any]  # Original ChromaDB response
    document_count: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "my_collection",
                "query": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence...\nDeep learning uses neural networks...",
                "documents": [
                    "Machine learning is a subset of artificial intelligence...",
                    "Deep learning uses neural networks..."
                ],
                "raw_results": {
                    "ids": [["doc1", "doc2"]],
                    "documents": [["Machine learning...", "Deep learning..."]],
                    "metadatas": [[{"source": "book1"}, {"source": "book2"}]],
                    "distances": [[0.1, 0.2]]
                },
                "document_count": 2
            }
        }