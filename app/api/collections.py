"""
Collections API router for ChromaDB operations
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import logging

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    CreateCollectionRequest, CreateCollectionResponse,
    CollectionResponse, ErrorResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_chroma_manager() -> ChromaDBManager:
    """Dependency injection placeholder - will be overridden in main.py"""
    pass

@router.post("/", 
    response_model=CreateCollectionResponse,
    status_code=status.HTTP_201_CREATED,
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
        collection = Collection(name=request.name, metadata=request.metadata)
        result = await chroma_manager.create_collection(collection)
        
        return CreateCollectionResponse(
            name=result["name"],
            id=result["id"],
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
    summary="Get collection by name",
    description="Retrieve collection information by name"
)
async def get_collection(
    collection_name: str,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get a collection by name"""
    try:
        result = await chroma_manager.get_collection(collection_name)
        return CollectionResponse(
            name=result["name"],
            id=result["id"],
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
                id=col["id"],
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
        success = await chroma_manager.delete_collection(collection_name)
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