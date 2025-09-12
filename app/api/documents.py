"""
Documents API router for ChromaDB document operations
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional
import logging

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    AddDocumentsRequest, AddDocumentsResponse,
    GetDocumentsResponse, UpdateDocumentsRequest, UpdateDocumentsResponse,
    DeleteDocumentsRequest, DeleteDocumentsResponse,
    QueryCollectionRequest, QueryResponse, Document
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_chroma_manager() -> ChromaDBManager:
    """Dependency injection placeholder - will be overridden in main.py"""
    pass

@router.post("/{collection_name}",
    response_model=AddDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add documents to collection",
    description="Add multiple documents to a specified collection"
)
async def add_documents(
    collection_name: str,
    documents: List[Document],
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Add documents to a collection"""
    try:
        success = await chroma_manager.add_documents(collection_name, documents)
        
        return AddDocumentsResponse(
            success=success,
            message=f"Successfully added {len(documents)} documents to collection {collection_name}",
            count=len(documents)
        )
    except Exception as e:
        logger.error(f"Error adding documents to {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add documents: {str(e)}"
        )

@router.get("/{collection_name}",
    response_model=GetDocumentsResponse,
    summary="Get documents from collection",
    description="Retrieve documents from a collection with optional filtering"
)
async def get_documents(
    collection_name: str,
    ids: Optional[List[str]] = Query(None, description="Document IDs to retrieve"),
    limit: Optional[int] = Query(None, description="Maximum number of documents to return"),
    offset: Optional[int] = Query(None, description="Number of documents to skip"),
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get documents from a collection"""
    try:
        documents = await chroma_manager.get_documents(
            collection_name=collection_name,
            ids=ids,
            limit=limit,
            offset=offset
        )
        
        return GetDocumentsResponse(
            documents=documents,
            count=len(documents)
        )
    except Exception as e:
        logger.error(f"Error getting documents from {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get documents: {str(e)}"
        )

@router.put("/{collection_name}",
    response_model=UpdateDocumentsResponse,
    summary="Update documents in collection",
    description="Update existing documents in a collection"
)
async def update_documents(
    collection_name: str,
    documents: List[Document],
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Update documents in a collection"""
    try:
        success = await chroma_manager.update_documents(collection_name, documents)
        
        return UpdateDocumentsResponse(
            success=success,
            message=f"Successfully updated {len(documents)} documents in collection {collection_name}",
            updated_count=len(documents)
        )
    except Exception as e:
        logger.error(f"Error updating documents in {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update documents: {str(e)}"
        )

@router.delete("/{collection_name}",
    response_model=DeleteDocumentsResponse,
    summary="Delete documents from collection",
    description="Delete documents from a collection by IDs"
)
async def delete_documents(
    collection_name: str,
    ids: List[str],
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Delete documents from a collection"""
    try:
        success = await chroma_manager.delete_documents(collection_name, ids)
        
        return DeleteDocumentsResponse(
            success=success,
            message=f"Successfully deleted {len(ids)} documents from collection {collection_name}",
            deleted_count=len(ids)
        )
    except Exception as e:
        logger.error(f"Error deleting documents from {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete documents: {str(e)}"
        )

@router.post("/{collection_name}/query",
    response_model=QueryResponse,
    summary="Query collection",
    description="Query documents in a collection using similarity search"
)
async def query_collection(
    collection_name: str,
    query_request: QueryCollectionRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Query documents in a collection"""
    try:
        # Override collection_name from the URL parameter
        query_request.collection_name = collection_name
        
        result = await chroma_manager.query_collection(collection_name, query_request.query)
        return result
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to query collection: {str(e)}"
        )

@router.get("/{collection_name}/count",
    summary="Get document count",
    description="Get the number of documents in a collection"
)
async def get_document_count(
    collection_name: str,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get document count for a collection"""
    try:
        collection_info = await chroma_manager.get_collection(collection_name)
        return {"collection_name": collection_name, "count": collection_info["count"]}
    except Exception as e:
        logger.error(f"Error getting document count for {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get document count: {str(e)}"
        )