"""
Documents API router for ChromaDB document operations
"""

from fastapi import APIRouter, HTTPException, Depends, status, Query, UploadFile, File, Form
from typing import List, Optional
import logging
from datetime import datetime
import uuid
import json

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    AddDocumentsResponse, GetDocumentsResponse, UpdateDocumentsRequest,
    UpdateDocumentsResponse, DeleteDocumentsRequest, DeleteDocumentsResponse,
    QueryCollectionRequest, QueryRequest, QueryResponse, Document, SimpleQueryRequest, SimpleQueryResponse
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
    description="Add documents to a specified collection via file upload or text string"
)
async def add_documents(
    collection_name: str,
    file: Optional[UploadFile] = File(None, description="Upload a text file (.txt, .csv, .json)"),
    text: Optional[str] = Form(None, description="Direct text content to add as a document"),
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Add documents to a collection from file upload or direct text string"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        docs_to_add = []

        # Handle file upload
        if file:
            if file.content_type not in ["text/plain", "text/csv", "application/json"]:
                raise ValueError("File must be a text file (.txt, .csv, or .json)")
            
            content = (await file.read()).decode('utf-8').strip()
            if not content:
                raise ValueError("Empty file content provided")
            
            # Try to parse as JSON if it's a .json file
            if file.filename and file.filename.endswith('.json'):
                try:
                    parsed_docs = json.loads(content)
                    if isinstance(parsed_docs, list):
                        for doc_data in parsed_docs:
                            if isinstance(doc_data, dict) and 'content' in doc_data:
                                meta = auto_generate_metadata(len(doc_data['content']), source="json_file")
                                docs_to_add.append(Document(
                                    id=doc_data.get('id', str(uuid.uuid4())),
                                    content=doc_data['content'],
                                    metadata=meta
                                ))
                    else:
                        raise ValueError("JSON file must contain a list of documents")
                except json.JSONDecodeError:
                    # Treat as plain text if JSON parsing fails
                    meta = auto_generate_metadata(len(content), source="file_upload")
                    docs_to_add.append(Document(
                        id=str(uuid.uuid4()),
                        content=content,
                        metadata=meta
                    ))
            else:
                # Handle as plain text file
                meta = auto_generate_metadata(len(content), source="file_upload")
                docs_to_add.append(Document(
                    id=str(uuid.uuid4()),
                    content=content,
                    metadata=meta
                ))

        # Handle direct text input
        elif text:
            meta = auto_generate_metadata(len(text), source="text_input")
            docs_to_add.append(Document(
                id=str(uuid.uuid4()),
                content=text,
                metadata=meta
            ))

        else:
            raise ValueError("Either 'file' or 'text' parameter must be provided")

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

# Rest of the endpoints remain the same...
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
    

    # Add these simplified models to your models.py file



# Add this new endpoint to your documents router

@router.post("/{collection_name}/simple-query",
    response_model=SimpleQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Simple query collection",
    description="Query documents in a collection using just a query string"
)
async def simple_query_collection(
    collection_name: str,
    request: SimpleQueryRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Query documents in a collection with simplified interface"""
    try:
        processed_name = preprocess_collection_name(collection_name)
        
        # Create the internal QueryRequest object
        query_request = QueryRequest(
            query_texts=[request.query],  # Just the query text
            n_results=request.n_results,
            include=["metadatas", "documents", "distances"]  # Standard includes
        )
        
        result = await chroma_manager.query_collection(processed_name, query_request)
        
        # Simplify the response format
        simplified_results = []
        if result.ids and len(result.ids) > 0:
            for i in range(len(result.ids[0])):  # ChromaDB returns nested lists
                doc_result = {
                    "id": result.ids[0][i],
                    "content": result.documents[0][i] if result.documents else None,
                    "metadata": result.metadatas[0][i] if result.metadatas else None,
                    "distance": result.distances[0][i] if result.distances else None
                }
                simplified_results.append(doc_result)
        
        return SimpleQueryResponse(
            results=simplified_results,
            count=len(simplified_results)
        )
    except Exception as e:
        logger.error(f"Error querying collection {collection_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to query collection: {str(e)}"
        )