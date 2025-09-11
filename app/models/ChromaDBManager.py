
# Enhanced ChromaDB management with better search capabilities
class ChromaDBManager:
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = SentenceTransformerEmbeddingFunction(embedding_model)
    
    def get_collection(self, avatar: str):
        """Get or create collection with proper error handling"""
        collection_name = f"documents_{avatar}"
        try:
            return self.client.get_collection(name=collection_name)
        except:
            return self.client.create_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
    
    def semantic_search(self, avatar: str, query: str, max_results: int = 3, 
                       similarity_threshold: float = 0.7) -> Dict:
        """Enhanced semantic search with similarity scoring"""
        try:
            collection = self.get_collection(avatar)
            results = collection.query(
                query_texts=[query], 
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if (results and results["documents"] and results["metadatas"] and 
                len(results["documents"]) > 0 and len(results["documents"][0]) > 0):
                
                context_parts = []
                docs = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results.get("distances", [[]])[0] if "distances" in results else []
                
                for i, (doc, metadata) in enumerate(zip(docs, metadatas)):
                    similarity = 1 - distances[i] if i < len(distances) else 1.0
                    
                    if similarity >= similarity_threshold:
                        context_parts.append({
                            "content": doc,
                            "source": metadata.get('source', 'unknown'),
                            "similarity": round(similarity, 3),
                            "metadata": metadata
                        })
                
                return {
                    "context_parts": context_parts,
                    "total_results": len(context_parts),
                    "query": query,
                    "has_results": len(context_parts) > 0
                }
            
            return {"context_parts": [], "total_results": 0, "query": query, "has_results": False}
            
        except Exception as e:
            logger.warning(f"Error retrieving context for {avatar}: {str(e)}")
            return {"context_parts": [], "total_results": 0, "query": query, "has_results": False}

    def get_collection_stats(self, avatar: str) -> Dict:
        """Get statistics about a collection"""
        try:
            collection = self.get_collection(avatar)
            count = collection.count()
            return {
                "document_count": count,
                "avatar": avatar,
                "collection_name": f"documents_{avatar}"
            }
        except Exception as e:
            return {
                "document_count": 0,
                "avatar": avatar,
                "error": str(e)
            }
        
    def ensure_collection_exists(self, avatar: str) -> tuple[bool, dict]:
        """Ensure collection exists for avatar, return (exists, stats)"""
        try:
            collection = self.get_collection(avatar)
            count = collection.count()
            stats = {
                "collection_exists": True,
                "document_count": count,
                "has_documents": count > 0,
                "avatar": avatar
            }
            logger.info(f"ChromaDB collection for {avatar}: {count} documents")
            return True, stats
        except Exception as e:
            logger.warning(f"Issue with ChromaDB collection for {avatar}: {str(e)}")
            # Try to create collection
            try:
                collection = self.client.create_collection(
                    name=f"documents_{avatar}",
                    embedding_function=self.embedding_function
                )
                stats = {
                    "collection_exists": True,
                    "document_count": 0,
                    "has_documents": False,
                    "avatar": avatar,
                    "created_new": True
                }
                return True, stats
            except Exception as create_error:
                logger.error(f"Failed to create collection for {avatar}: {str(create_error)}")
                return False, {"collection_exists": False, "error": str(create_error)}

