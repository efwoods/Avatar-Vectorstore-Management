# models.py - Pydantic models for API requests/responses
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CreateCollectionRequest(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = None

class AddRequest(BaseModel):
    ids: List[str]
    documents: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None

class UpsertRequest(AddRequest):
    pass

class DeleteRequest(BaseModel):
    ids: List[str]

class GetRequest(BaseModel):
    ids: Optional[List[str]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    include: List[str] = ["metadatas", "documents", "embeddings"]

class QueryRequest(BaseModel):
    query_texts: List[str]
    n_results: Optional[int] = 10
    include: List[str] = ["metadatas", "documents", "embeddings"]