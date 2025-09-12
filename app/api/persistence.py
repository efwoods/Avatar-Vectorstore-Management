"""
Persistence API router for S3 backup and restore operations
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import logging

from db.ChromaDBManager import ChromaDBManager
from db.schema.models import (
    BackupRequest, BackupResponse,
    RestoreRequest, RestoreResponse,
    ListBackupsRequest, ListBackupsResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

async def get_chroma_manager() -> ChromaDBManager:
    """Dependency injection placeholder - will be overridden in main.py"""
    pass

@router.post("/backup",
    response_model=BackupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Backup vectorstore to S3",
    description="Create a backup of the ChromaDB vectorstore and upload it to S3"
)
async def backup_to_s3(
    request: BackupRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Backup ChromaDB vectorstore to S3"""
    try:
        backup_info = await chroma_manager.backup_to_s3(
            user_id=request.user_id,
            avatar_id=request.avatar_id
        )
        
        return BackupResponse(
            success=True,
            message=f"Successfully backed up vectorstore for user {request.user_id}",
            backup_info=backup_info
        )
    except Exception as e:
        logger.error(f"Error backing up to S3: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to backup vectorstore: {str(e)}"
        )

@router.post("/restore",
    response_model=RestoreResponse,
    summary="Restore vectorstore from S3",
    description="Restore ChromaDB vectorstore from an S3 backup"
)
async def restore_from_s3(
    request: RestoreRequest,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Restore ChromaDB vectorstore from S3"""
    try:
        success = await chroma_manager.restore_from_s3(
            user_id=request.user_id,
            backup_timestamp=request.backup_timestamp,
            avatar_id=request.avatar_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to restore vectorstore"
            )
        
        return RestoreResponse(
            success=True,
            message=f"Successfully restored vectorstore for user {request.user_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring from S3: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore vectorstore: {str(e)}"
        )

@router.get("/backups",
    response_model=ListBackupsResponse,
    summary="List available backups",
    description="List all available backups for a user in S3"
)
async def list_backups(
    user_id: str,
    avatar_id: str = None,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """List available S3 backups"""
    try:
        backups = await chroma_manager.list_s3_backups(
            user_id=user_id,
            avatar_id=avatar_id
        )
        
        return ListBackupsResponse(
            backups=backups,
            count=len(backups)
        )
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list backups: {str(e)}"
        )

@router.delete("/backups",
    summary="Delete backup from S3",
    description="Delete a specific backup from S3"
)
async def delete_backup(
    user_id: str,
    backup_timestamp: str,
    avatar_id: str = None,
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Delete a backup from S3"""
    try:
        # Determine S3 path
        if avatar_id:
            s3_path = chroma_manager.settings.get_s3_avatar_vectorstore_path(user_id, avatar_id)
        else:
            s3_path = chroma_manager.settings.get_s3_vectorstore_path(user_id)
        
        # Delete backup file
        backup_key = f"{s3_path}backup_{backup_timestamp}.zip"
        metadata_key = f"{s3_path}metadata_{backup_timestamp}.json"
        
        # Delete from S3
        chroma_manager.s3_client.delete_object(
            Bucket=chroma_manager.settings.s3_bucket_name,
            Key=backup_key
        )
        chroma_manager.s3_client.delete_object(
            Bucket=chroma_manager.settings.s3_bucket_name,
            Key=metadata_key
        )
        
        return {"success": True, "message": f"Successfully deleted backup {backup_timestamp}"}
    except Exception as e:
        logger.error(f"Error deleting backup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete backup: {str(e)}"
        )

@router.get("/status",
    summary="Get persistence status",
    description="Get the status of S3 connectivity and backup information"
)
async def get_persistence_status(
    chroma_manager: ChromaDBManager = Depends(get_chroma_manager)
):
    """Get persistence status"""
    try:
        # Test S3 connectivity
        s3_status = "connected"
        try:
            chroma_manager.s3_client.head_bucket(Bucket=chroma_manager.settings.s3_bucket_name)
        except Exception:
            s3_status = "disconnected"
        
        # Get local vectorstore info
        collections = await chroma_manager.list_collections()
        
        return {
            "s3_status": s3_status,
            "s3_bucket": chroma_manager.settings.s3_bucket_name,
            "local_collections_count": len(collections),
            "local_collections": [col["name"] for col in collections],
            "persistence_directory": chroma_manager.settings.chroma_persist_directory
        }
    except Exception as e:
        logger.error(f"Error getting persistence status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get persistence status: {str(e)}"
        )