"""
Configuration settings for the ChromaDB Vectorstore Service
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    app_name: str = "ChromaDB Vectorstore Service"
    version: str = "1.0.0"
    debug: bool = False
    port: int = 8000
    
    # ChromaDB settings
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_directory: str = "./chroma_data"
    
    # S3 settings
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    s3_bucket_name: str
    s3_prefix: str = "vectorstore"
    
    # Security settings
    api_key: Optional[str] = None
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_batch_size: int = 100
    
    # Backup settings
    auto_backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_s3_vectorstore_path(self, user_id: str) -> str:
        """Get S3 path for user vectorstore"""
        return f"users/{user_id}/vectorstore/"
    
    def get_s3_avatar_vectorstore_path(self, user_id: str, avatar_id: str) -> str:
        """Get S3 path for avatar vectorstore data"""
        return f"users/{user_id}/avatars/{avatar_id}/vectorstore_data/"
    
    def get_s3_avatar_metadata_path(self, user_id: str, avatar_id: str) -> str:
        """Get S3 path for avatar vectorstore metadata"""
        return f"users/{user_id}/avatars/{avatar_id}/vectorstore_metadata/"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()