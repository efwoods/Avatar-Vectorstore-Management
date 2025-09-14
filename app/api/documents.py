"""
Documents API router for ChromaDB document operations
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query, UploadFile, File
from typing import List, Optional
import logging
from datetime import datetime
import uuid
import json

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    AddDocumentsResponse, GetDocumentsResponse, UpdateDocumentsRequest,
    UpdateDocumentsResponse, DeleteDocumentsRequest, DeleteDocumentsResponse,
    QueryCollectionRequest, QueryResponse, Document
)

logger = logging.getLogger(__name__)
router = APIRouter()

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

def auto_generate_metadata(content_length: int, source: str = "document") -> dict:
    """Automatically generate non-empty metadata for a document"""
    return {
        "created_at": datetime.now().isoformat(),
        "doc_id": str(uuid.uuid4()),
        "type": "document",
        "source": source,
        "length": content_length
    }

@router.post("/{collection_name}",
    response_model=AddDocumentsResponse,
    status_code=status.HTTP_200_OK,
    summary="Add documents to collection",
    description="Add documents to a specified collection, accepting either JSON documents or a text file"
)
async def add_documents(
    collection_name: str,
    documents: Optional[List[Document]] = None,
    file: Optional[UploadFile] = File(None),
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Add documents to a collection from JSON or text file"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        docs_to_add = []

        # Handle JSON documents
        if documents:
            for doc in documents:
                meta = auto_generate_metadata(len(doc.content), source="json_input")
                docs_to_add.append(Document(
                    id=doc.id or str(uuid.uuid4()),
                    content=doc.content,
                    metadata=meta
                ))

        # Handle file upload
        if file:
            if file.content_type not in ["text/plain", "text/csv"]:
                raise ValueError("File must be a text file (.txt or .csv)")
            content = (await file.read()).decode('utf-8').strip()
            if not content:
                raise ValueError("Empty file content provided")
            docs_to_add.append(Document(
                id=str(uuid.uuid4()),
                content=content,
                metadata=auto_generate_metadata(len(content), source="file_upload")
            ))

        if not docs_to_add:
            raise ValueError("No documents or file provided")

        success = await chroma_manager.add_documents(processed_name, docs_to_add)
        
        return AddDocumentsResponse(
            success=success,
            message=f"Successfully added {len(docs_to_add)} documents to collection {processed_name}",
            count=len(docs_to_add)
        )
    except Exception as e:
        logger.error(f"Error adding documents to {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to add documents: {str(e)}"
        )

@router.get("/{collection_name}",
    response_model=GetDocumentsResponse,
    status_code=status.HTTP_200_OK,
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
        processed_name = preprocess_collection_name(collection_name)
        documents = await chroma_manager.get_documents(
            collection_name=processed_name,
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
    status_code=status.HTTP_200_OK,
    summary="Update documents in collection",
    description="Update existing documents in a collection"
)
async def update_documents(
    collection_name: str,
    documents: List[UpdateDocumentsRequest],
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Update documents in a collection"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        docs_with_meta = []
        for doc in documents:
            meta = auto_generate_metadata(len(doc.content))
            docs_with_meta.append(Document(id=doc.id, content=doc.content, metadata=meta))
        success = await chroma_manager.update_documents(processed_name, docs_with_meta)
        
        return UpdateDocumentsResponse(
            success=success,
            message=f"Successfully updated {len(documents)} documents in collection {processed_name}",
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
    status_code=status.HTTP_200_OK,
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
        processed_name = preprocess_collection_name(collection_name)
        success = await chroma_manager.delete_documents(processed_name, ids)
        
        return DeleteDocumentsResponse(
            success=success,
            message=f"Successfully deleted {len(ids)} documents from collection {processed_name}",
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
    status_code=status.HTTP_200_OK,
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
        processed_name = preprocess_collection_name(collection_name)
        query_request.collection_name = processed_name
        
        result = await chroma_manager.query_collection(processed_name, query_request.query)
        return result
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to query collection: {str(e)}"
        )

@router.get("/{collection_name}/count",
    status_code=status.HTTP_200_OK,
    summary="Get document count",
    description="Get the number of documents in a collection"
)
async def get_document_count(
    collection_name: str,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get document count for a collection"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        collection_info = await chroma_manager.get_collection(processed_name)
        return {"collection_name": processed_name, "count": collection_info["count"]}
    except Exception as e:
        logger.error(f"Error getting document count for {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to get document count: {str(e)}"
        )


## Changes Explained
# - **Unified Endpoint**: Modified `add_documents` to accept either `List[Document]` or `UploadFile`, removing `embedding` from `Document` model. Supports JSON documents or text file uploads (`.txt`, `.csv`).
# - **File Handling**: Reads file content, creates `Document` with auto-generated metadata (`source: file_upload`). ChromaDB auto-embeds content.
# - **Metadata**: Updated `auto_generate_metadata` to include `source` (json_input or file_upload) and content length.
# - **Curl Example**: Use `-F 'file=@content.txt'` for file uploads.
# - **Frontend**: Added React component for file uploads using `FormData`.
# - **Why**: Simplifies API by removing embedding parameter, supports raw text files, and ensures consistent metadata and embedding by ChromaDB.

## Changes Explained

# - **Name Preprocessing**: Added `preprocess_collection_name` function to the documents router, matching the logic from the collections router (lowercase, replace spaces/hyphens with underscores, keep alphanumeric and underscores, validate non-empty). Applied to `collection_name` in all endpoints: `add_documents`, `get_documents`, `update_documents`, `delete_documents`, `query_collection`, and `get_document_count`.

# - **Status Code Update**: Changed `status_code` for `add_documents`, `get_documents`, `update_documents`, `delete_documents`, `query_collection`, and `get_document_count` to `status.HTTP_200_OK` (200) for successful responses, aligning with the collections router's convention (except for DELETE, which now returns 200 with a response body instead of 204, per consistency request).

# - **Message Updates**: Updated response messages in `add_documents`, `update_documents`, and `delete_documents` to use `processed_name` for consistency in logging and responses.

# - **Why These Changes**: Ensures consistent name preprocessing across all document endpoints, preventing errors from invalid collection names (e.g., spaces, special characters) in ChromaDB. Aligns status codes with the 200 OK requirement for successful responses, maintaining REST API consistency across routers.