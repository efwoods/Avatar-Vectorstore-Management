"""
Collections API router for ChromaDB operations
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import logging
import uuid
from datetime import datetime

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    CreateCollectionRequest, CreateCollectionResponse,
    CollectionResponse, ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Import the dependency function from main module at runtime
def get_chroma_manager() -> ChromaDBManager:
    """Get the chroma manager instance from main module"""
    from main import get_chroma_manager as _get_chroma_manager
    return _get_chroma_manager()

def preprocess_collection_name(name: str) -> str:
    """Preprocess collection name to ensure validity"""
    processed_name = name.lower().replace(' ', '_').replace('-', '_')
    processed_name = ''.join(c for c in processed_name if c.isalnum() or c == '_')
    if not processed_name:
        raise ValueError("Invalid collection name after preprocessing")
    return processed_name

@router.post("/", 
    response_model=CreateCollectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a new collection",
    description="Create a new ChromaDB collection with optional metadata"
)
async def create_collection(
    request: CreateCollectionRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Create a new collection"""
    try:
        from db.schema.models import Collection
        processed_name = preprocess_collection_name(request.name)
        
        # Ensure metadata is always sent
        metadata = request.metadata or {}
        if not metadata:
            metadata = {"created_by": chroma_manager.user_id, "created_at": datetime.now().isoformat()}
        
        collection = Collection(name=processed_name, metadata=metadata)
        result = await chroma_manager.create_collection(collection)
        
        return CreateCollectionResponse(
            name=result["name"],
            id=str(result["id"]),  # Convert UUID to string
            metadata=result["metadata"],
            count=result["count"]
        )
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create collection: {str(e)}"
        )

@router.get("/{collection_name}",
    response_model=CollectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get collection by name",
    description="Retrieve collection information by name"
)
async def get_collection(
    collection_name: str,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get a collection by name"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        result = await chroma_manager.get_collection(processed_name)
        return CollectionResponse(
            name=result["name"],
            id=str(result["id"]),  # Convert UUID to string
            metadata=result["metadata"],
            count=result["count"]
        )
    except Exception as e:
        logger.error(f"Error getting collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection not found: {str(e)}"
        )

@router.get("/",
    response_model=List[CollectionResponse],
    status_code=status.HTTP_200_OK,
    summary="List all collections",
    description="Get a list of all collections in the vectorstore"
)
async def list_collections(
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """List all collections"""
    try:
        collections = await chroma_manager.list_collections()
        return [
            CollectionResponse(
                name=col["name"],
                id=str(col["id"]),  # Convert UUID to string
                metadata=col["metadata"],
                count=col["count"]
            )
            for col in collections
        ]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )

@router.delete("/{collection_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete collection",
    description="Delete a collection by name"
)
async def delete_collection(
    collection_name: str,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Delete a collection"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        success = await chroma_manager.delete_collection(processed_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Collection not found"
            )
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to delete collection: {str(e)}"
        )

# - **Status Code Update**: Changed `status_code` for `create_collection`, `get_collection`, and `list_collections` from `HTTP_201_CREATED` (201) and implicit 200 to explicit `status.HTTP_200_OK` (200) for successful responses, per request. `delete_collection` remains `HTTP_204_NO_CONTENT` (204) as it returns no content, which is standard for DELETE operations.

# - **Why These Changes**: Aligns API with request for 200 OK status on successful operations for POST and GET endpoints, ensuring consistency. 204 for DELETE is retained as it follows REST conventions for operations that don’t return a response body.