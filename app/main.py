# app.py - FastAPI application
from fastapi import FastAPI, HTTPException
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os
from typing import List

from models import (
    CreateCollectionRequest, AddRequest, UpsertRequest,
    DeleteRequest, GetRequest, QueryRequest
)

app = FastAPI(title="Chroma DB API", version="1.0.0")

PERSISTENT_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
client = PersistentClient(path=PERSISTENT_PATH)

# Global embedding model (shared for all collections)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding_function():
    """Returns the embedding function for Chroma collections."""
    def embed(texts: List[str]) -> List[List[float]]:
        return embedding_model.encode(texts).tolist()
    return embed

@app.post("/collections/create")
def create_collection(req: CreateCollectionRequest):
    """Create a new collection with embedding function."""
    try:
        embedding_fn = get_embedding_function()
        client.create_collection(
            name=req.name,
            metadata=req.metadata or {},
            embedding_function=embedding_fn
        )
        return {"status": "created", "name": req.name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/collections/list")
def list_collections():
    """List all collections."""
    try:
        res = client.list_collections()
        return {"names": [c.name for c in res]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/collections/{name}")
def delete_collection(name: str):
    """Delete a collection."""
    try:
        client.delete_collection(name)
        return {"status": "deleted", "name": name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/collections/{name}/add")
def add_to_collection(name: str, req: AddRequest):
    """Add data to collection (embeds documents if no embeddings)."""
    try:
        coll = client.get_or_create_collection(name)  # Use get_or_create for safety
        coll.add(
            ids=req.ids,
            documents=req.documents,
            embeddings=req.embeddings,
            metadatas=req.metadatas
        )
        return {"status": "added", "count": len(req.ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{name}/upsert")
def upsert_to_collection(name: str, req: UpsertRequest):
    """Upsert data (update if ID exists, add otherwise)."""
    try:
        coll = client.get_or_create_collection(name)
        coll.upsert(
            ids=req.ids,
            documents=req.documents,
            embeddings=req.embeddings,
            metadatas=req.metadatas
        )
        return {"status": "upserted", "count": len(req.ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{name}/delete")
def delete_from_collection(name: str, req: DeleteRequest):
    """Delete data by IDs."""
    try:
        coll = client.get_collection(name)
        coll.delete(ids=req.ids)
        return {"status": "deleted", "count": len(req.ids)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{name}/get")
def get_from_collection(name: str, req: GetRequest):
    """Get data by IDs or paginated."""
    try:
        coll = client.get_collection(name)
        res = coll.get(
            ids=req.ids,
            limit=req.limit,
            offset=req.offset,
            include=req.include
        )
        return {
            "ids": res["ids"],
            "documents": res.get("documents"),
            "metadatas": res.get("metadatas"),
            "embeddings": res.get("embeddings")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/collections/{name}/query")
def query_collection(name: str, req: QueryRequest):
    """Query collection for similar items."""
    try:
        coll = client.get_collection(name)
        res = coll.query(
            query_texts=req.query_texts,
            n_results=req.n_results,
            include=req.include
        )
        # For single query, flatten lists
        if len(req.query_texts) == 1:
            res = {k: v[0] if isinstance(v, list) and len(v) > 0 else v for k, v in res.items()}
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)