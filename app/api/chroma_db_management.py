
# from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
# from fastapi.responses import HTMLResponse, JSONResponse
# from typing import List, Optional, Dict, Any
# from pydantic import BaseModel
# import uuid
# import time
# import json
# import logging
# from pathlib import Path
# import asyncio
# from datetime import datetime

# # Pydantic Models
# class DocumentMetadata(BaseModel):
#     source: Optional[str] = None
#     title: Optional[str] = None
#     author: Optional[str] = None
#     category: Optional[str] = None
#     tags: Optional[List[str]] = []
#     custom_fields: Optional[Dict[str, Any]] = {}

# class DocumentRequest(BaseModel):
#     content: str
#     source: str
#     metadata: Optional[Dict[str, Any]] = {}

# class BulkDocumentRequest(BaseModel):
#     documents: List[DocumentRequest]

# class DocumentResponse(BaseModel):
#     status: str
#     document_id: str
#     avatar: str
#     filename: Optional[str] = None
#     file_size: Optional[int] = None
#     content_preview: Optional[str] = None

# class BulkDocumentResponse(BaseModel):
#     status: str
#     document_ids: List[str]
#     avatar: str
#     processed_count: int
#     failed_count: int
#     errors: Optional[List[str]] = []

# # File processing utilities
# class DocumentProcessor:
#     SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.json', '.csv', '.xml', '.html'}
#     MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
#     @staticmethod
#     async def read_file_content(file: UploadFile) -> str:
#         """Read and extract text content from uploaded file"""
#         content = await file.read()
#         file_extension = Path(file.filename).suffix.lower()
        
#         if file_extension == '.txt' or file_extension == '.md':
#             return content.decode('utf-8')
#         elif file_extension == '.json':
#             return content.decode('utf-8')
#         elif file_extension == '.csv':
#             return content.decode('utf-8')
#         elif file_extension == '.html' or file_extension == '.xml':
#             return content.decode('utf-8')
#         elif file_extension == '.pdf':
#             # For PDF, you'd need additional libraries like PyPDF2 or pdfplumber
#             # This is a placeholder - implement PDF extraction based on your needs
#             return f"PDF content extraction needed for: {file.filename}"
#         elif file_extension == '.docx':
#             # For DOCX, you'd need python-docx library
#             # This is a placeholder - implement DOCX extraction based on your needs
#             return f"DOCX content extraction needed for: {file.filename}"
#         else:
#             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
    
#     @staticmethod
#     def validate_file(file: UploadFile) -> bool:
#         """Validate file type and size"""
#         if not file.filename:
#             return False
            
#         file_extension = Path(file.filename).suffix.lower()
#         if file_extension not in DocumentProcessor.SUPPORTED_EXTENSIONS:
#             return False
            
#         return True
    
#     @staticmethod
#     def create_metadata(file: UploadFile, custom_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
#         """Create comprehensive metadata for document"""
#         metadata = {
#             "filename": file.filename,
#             "file_size": file.size if hasattr(file, 'size') else 0,
#             "content_type": file.content_type,
#             "upload_timestamp": datetime.now().isoformat(),
#             "file_extension": Path(file.filename).suffix.lower() if file.filename else "",
#         }
        
#         if custom_metadata:
#             metadata.update(custom_metadata)
            
#         return metadata

# # FastAPI Endpoints
# @app.post("/upload_document", 
#           response_model=DocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_single_document(
#     file: UploadFile = File(...),
#     source: Optional[str] = Form(None),
#     title: Optional[str] = Form(None),
#     author: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload a single document to ChromaDB with metadata"""
#     try:
#         # Validate file
#         if not DocumentProcessor.validate_file(file):
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Unsupported file type. Supported types: {', '.join(DocumentProcessor.SUPPORTED_EXTENSIONS)}"
#             )
        
#         # Check file size
#         if hasattr(file, 'size') and file.size > DocumentProcessor.MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File too large. Maximum size: {DocumentProcessor.MAX_FILE_SIZE / (1024*1024)}MB"
#             )
        
#         # Read file content
#         content = await DocumentProcessor.read_file_content(file)
        
#         # Parse custom metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         # Parse tags
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         # Create comprehensive metadata
#         file_metadata = DocumentProcessor.create_metadata(file, parsed_custom_metadata)
        
#         # Add form data to metadata
#         if source:
#             file_metadata["source"] = source
#         if title:
#             file_metadata["title"] = title
#         if author:
#             file_metadata["author"] = author
#         if category:
#             file_metadata["category"] = category
#         if parsed_tags:
#             file_metadata["tags"] = parsed_tags
        
#         # Add to ChromaDB
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
#         doc_id = str(uuid.uuid4())
        
#         chroma_metadata = {
#             "source": source or file.filename,
#             "avatar": model_manager.current_avatar,
#             "created_at": time.time(),
#             "document_id": doc_id,
#             **file_metadata
#         }
        
#         collection.add(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[chroma_metadata]
#         )
        
#         # Broadcast update
#         await websocket_manager.broadcast({
#             "type": "document_uploaded",
#             "avatar": model_manager.current_avatar,
#             "document_id": doc_id,
#             "filename": file.filename
#         })
        
#         return DocumentResponse(
#             status="Document uploaded successfully",
#             document_id=doc_id,
#             avatar=model_manager.current_avatar,
#             filename=file.filename,
#             file_size=file.size if hasattr(file, 'size') else 0,
#             content_preview=content[:200] + "..." if len(content) > 200 else content
#         )
        
#     except Exception as e:
#         logger.error(f"Error uploading document: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# @app.post("/upload_documents_bulk",
#           response_model=BulkDocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_multiple_documents(
#     files: List[UploadFile] = File(...),
#     source: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload multiple documents to ChromaDB in bulk"""
#     try:
#         if len(files) > 50:  # Limit bulk uploads
#             raise HTTPException(status_code=400, detail="Maximum 50 files allowed per bulk upload")
        
#         # Parse common metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         documents = []
#         doc_ids = []
#         metadatas = []
#         processed_count = 0
#         failed_count = 0
#         errors = []
        
#         for file in files:
#             try:
#                 # Validate each file
#                 if not DocumentProcessor.validate_file(file):
#                     errors.append(f"Unsupported file type: {file.filename}")
#                     failed_count += 1
#                     continue
                
#                 # Read content
#                 content = await DocumentProcessor.read_file_content(file)
                
#                 # Create metadata
#                 file_metadata = DocumentProcessor.create_metadata(file, parsed_custom_metadata)
                
#                 # Add form data to metadata
#                 if source:
#                     file_metadata["source"] = source
#                 if category:
#                     file_metadata["category"] = category
#                 if parsed_tags:
#                     file_metadata["tags"] = parsed_tags
                
#                 doc_id = str(uuid.uuid4())
                
#                 chroma_metadata = {
#                     "source": source or file.filename,
#                     "avatar": model_manager.current_avatar,
#                     "created_at": time.time(),
#                     "document_id": doc_id,
#                     **file_metadata
#                 }
                
#                 documents.append(content)
#                 doc_ids.append(doc_id)
#                 metadatas.append(chroma_metadata)
#                 processed_count += 1
                
#             except Exception as e:
#                 errors.append(f"Error processing {file.filename}: {str(e)}")
#                 failed_count += 1
#                 logger.error(f"Error processing file {file.filename}: {str(e)}")
        
#         # Add all valid documents to ChromaDB
#         if documents:
#             collection.add(
#                 documents=documents,
#                 ids=doc_ids,
#                 metadatas=metadatas
#             )
            
#             # Broadcast update
#             await websocket_manager.broadcast({
#                 "type": "documents_bulk_uploaded",
#                 "avatar": model_manager.current_avatar,
#                 "count": processed_count,
#                 "failed_count": failed_count
#             })
        
#         return BulkDocumentResponse(
#             status=f"Bulk upload completed. {processed_count} successful, {failed_count} failed",
#             document_ids=doc_ids,
#             avatar=model_manager.current_avatar,
#             processed_count=processed_count,
#             failed_count=failed_count,
#             errors=errors if errors else None
#         )
        
#     except Exception as e:
#         logger.error(f"Error in bulk document upload: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error in bulk upload: {str(e)}")

# @app.get("/supported_formats", tags=["ChromaDB Document Management"])
# async def get_supported_formats():
#     """Get list of supported file formats"""
#     return {
#         "supported_extensions": list(DocumentProcessor.SUPPORTED_EXTENSIONS),
#         "max_file_size_mb": DocumentProcessor.MAX_FILE_SIZE / (1024*1024),
#         "max_bulk_files": 50
#     }

# @app.post("/upload_text_content",
#           response_model=DocumentResponse,
#           tags=["ChromaDB Document Management"])
# async def upload_text_content(
#     content: str = Form(...),
#     source: str = Form(...),
#     title: Optional[str] = Form(None),
#     author: Optional[str] = Form(None),
#     category: Optional[str] = Form(None),
#     tags: Optional[str] = Form(None),
#     custom_metadata: Optional[str] = Form(None)
# ):
#     """Upload raw text content directly (no file)"""
#     try:
#         # Parse custom metadata
#         parsed_custom_metadata = {}
#         if custom_metadata:
#             try:
#                 parsed_custom_metadata = json.loads(custom_metadata)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in custom_metadata")
        
#         # Parse tags
#         parsed_tags = []
#         if tags:
#             parsed_tags = [tag.strip() for tag in tags.split(',')]
        
#         # Create metadata
#         doc_metadata = {
#             "upload_timestamp": datetime.now().isoformat(),
#             "content_type": "text/plain",
#             "content_length": len(content),
#             **parsed_custom_metadata
#         }
        
#         # Add form data to metadata
#         if title:
#             doc_metadata["title"] = title
#         if author:
#             doc_metadata["author"] = author
#         if category:
#             doc_metadata["category"] = category
#         if parsed_tags:
#             doc_metadata["tags"] = parsed_tags
        
#         # Add to ChromaDB
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
#         doc_id = str(uuid.uuid4())
        
#         chroma_metadata = {
#             "source": source,
#             "avatar": model_manager.current_avatar,
#             "created_at": time.time(),
#             "document_id": doc_id,
#             **doc_metadata
#         }
        
#         collection.add(
#             documents=[content],
#             ids=[doc_id],
#             metadatas=[chroma_metadata]
#         )
        
#         # Broadcast update
#         await websocket_manager.broadcast({
#             "type": "text_content_uploaded",
#             "avatar": model_manager.current_avatar,
#             "document_id": doc_id,
#             "source": source
#         })
        
#         return DocumentResponse(
#             status="Text content uploaded successfully",
#             document_id=doc_id,
#             avatar=model_manager.current_avatar,
#             content_preview=content[:200] + "..." if len(content) > 200 else content
#         )
        
#     except Exception as e:
#         logger.error(f"Error uploading text content: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error uploading text content: {str(e)}")

# @app.get("/upload_form", response_class=HTMLResponse, tags=["ChromaDB Document Management"])
# async def get_upload_form():
#     """Serve HTML form for document upload"""
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>ChromaDB Document Upload</title>
#         <style>
#             body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
#             .form-group { margin-bottom: 15px; }
#             label { display: block; margin-bottom: 5px; font-weight: bold; }
#             input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
#             button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
#             button:hover { background: #45a049; }
#             .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 4px; }
#             .file-list { margin-top: 10px; }
#             .file-item { background: #f5f5f5; padding: 5px 10px; margin: 5px 0; border-radius: 3px; }
#         </style>
#     </head>
#     <body>
#         <h1>ChromaDB Document Upload</h1>
        
#         <h2>Single Document Upload</h2>
#         <form id="singleForm" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label>File:</label>
#                 <input type="file" name="file" required accept=".txt,.md,.pdf,.docx,.json,.csv,.xml,.html">
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" placeholder="Document source">
#             </div>
#             <div class="form-group">
#                 <label>Title:</label>
#                 <input type="text" name="title" placeholder="Document title">
#             </div>
#             <div class="form-group">
#                 <label>Author:</label>
#                 <input type="text" name="author" placeholder="Document author">
#             </div>
#             <div class="form-group">
#                 <label>Category:</label>
#                 <input type="text" name="category" placeholder="Document category">
#             </div>
#             <div class="form-group">
#                 <label>Tags (comma-separated):</label>
#                 <input type="text" name="tags" placeholder="tag1, tag2, tag3">
#             </div>
#             <button type="submit">Upload Single Document</button>
#         </form>
        
#         <hr style="margin: 40px 0;">
        
#         <h2>Bulk Document Upload</h2>
#         <form id="bulkForm" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label>Files (multiple):</label>
#                 <input type="file" name="files" multiple required accept=".txt,.md,.pdf,.docx,.json,.csv,.xml,.html">
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" placeholder="Common source for all documents">
#             </div>
#             <div class="form-group">
#                 <label>Category:</label>
#                 <input type="text" name="category" placeholder="Common category">
#             </div>
#             <div class="form-group">
#                 <label>Tags (comma-separated):</label>
#                 <input type="text" name="tags" placeholder="tag1, tag2, tag3">
#             </div>
#             <button type="submit">Upload Multiple Documents</button>
#         </form>
        
#         <hr style="margin: 40px 0;">
        
#         <h2>Text Content Upload</h2>
#         <form id="textForm">
#             <div class="form-group">
#                 <label>Content:</label>
#                 <textarea name="content" rows="8" required placeholder="Enter your text content here..."></textarea>
#             </div>
#             <div class="form-group">
#                 <label>Source:</label>
#                 <input type="text" name="source" required placeholder="Content source">
#             </div>
#             <div class="form-group">
#                 <label>Title:</label>
#                 <input type="text" name="title" placeholder="Content title">
#             </div>
#             <div class="form-group">
#                 <label>Author:</label>
#                 <input type="text" name="author" placeholder="Content author">
#             </div>
#             <button type="submit">Upload Text Content</button>
#         </form>
        
#         <div id="results" style="margin-top: 20px;"></div>
        
#         <script>
#             function showResult(message, isError = false) {
#                 const results = document.getElementById('results');
#                 results.innerHTML = `<div style="padding: 10px; border-radius: 4px; background: ${isError ? '#ffebee' : '#e8f5e9'}; color: ${isError ? '#c62828' : '#2e7d32'};">${message}</div>`;
#             }
            
#             document.getElementById('singleForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_document', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status} (ID: ${result.document_id})`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
            
#             document.getElementById('bulkForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_documents_bulk', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status}`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
            
#             document.getElementById('textForm').addEventListener('submit', async (e) => {
#                 e.preventDefault();
#                 const formData = new FormData(e.target);
#                 try {
#                     const response = await fetch('/upload_text_content', { method: 'POST', body: formData });
#                     const result = await response.json();
#                     showResult(`Success: ${result.status} (ID: ${result.document_id})`);
#                 } catch (error) {
#                     showResult(`Error: ${error.message}`, true);
#                 }
#             });
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# # Additional utility endpoints
# @app.get("/documents/search", tags=["ChromaDB Document Management"])
# async def search_documents(
#     query: str,
#     limit: int = 10,
#     category: Optional[str] = None,
#     author: Optional[str] = None
# ):
#     """Search documents in ChromaDB with optional filters"""
#     try:
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         # Build where clause for filtering
#         where_clause = {"avatar": model_manager.current_avatar}
#         if category:
#             where_clause["category"] = category
#         if author:
#             where_clause["author"] = author
        
#         results = collection.query(
#             query_texts=[query],
#             n_results=limit,
#             where=where_clause if len(where_clause) > 1 else None
#         )
        
#         return {
#             "query": query,
#             "results": results,
#             "count": len(results.get('documents', []))
#         }
        
#     except Exception as e:
#         logger.error(f"Error searching documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

# @app.get("/documents/stats", tags=["ChromaDB Document Management"])
# async def get_document_stats():
#     """Get statistics about documents in ChromaDB"""
#     try:
#         collection = chroma_manager.get_collection(model_manager.current_avatar)
        
#         # Get all documents for this avatar
#         all_docs = collection.get(where={"avatar": model_manager.current_avatar})
        
#         total_docs = len(all_docs.get('documents', []))
        
#         # Analyze metadata
#         categories = {}
#         authors = {}
#         file_types = {}
        
#         for metadata in all_docs.get('metadatas', []):
#             if 'category' in metadata:
#                 categories[metadata['category']] = categories.get(metadata['category'], 0) + 1
#             if 'author' in metadata:
#                 authors[metadata['author']] = authors.get(metadata['author'], 0) + 1
#             if 'file_extension' in metadata:
#                 file_types[metadata['file_extension']] = file_types.get(metadata['file_extension'], 0) + 1
        
#         return {
#             "avatar": model_manager.current_avatar,
#             "total_documents": total_docs,
#             "categories": categories,
#             "authors": authors,
#             "file_types": file_types
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting document stats: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error getting document stats: {str(e)}")
