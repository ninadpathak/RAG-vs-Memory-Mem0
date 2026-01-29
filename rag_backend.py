import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

class RAGBackend:
    """
    A Production-grade RAG backend wrapper using ChromaDB.
    Simulates a standard 'Enterprise Search' setup with persistent storage,
    metadata filtering logic, and error handling.
    """
    
    def __init__(self, collection_name: str = "enterprise_knowledge_base"):
        # Use persistent storage to simulate real DB behavior, not just in-memory
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Reset for benchmark cleanliness (in real world, we wouldn't do this)
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Standard setting
        )
        
    def ingest_documents(self, docs: List[Dict[str, Any]]) -> None:
        """
        Ingests documents with rich metadata.
        Standard RAG relies heavily on metadata for filtering.
        """
        ids = [d["id"] for d in docs]
        contents = [d["content"] for d in docs]
        
        # Flatten metadata for Chroma (it prefers flat dicts)
        metadatas = []
        for d in docs:
            meta = {
                "topic": d["topic"],
                "department": d["department"],
                "status": d["status"],
                "source": d["metadata"]["source"],
                "version": d["metadata"]["version"]
            }
            metadatas.append(meta)
            
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
    def search(self, query: str, user_context: Optional[Dict] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Performs vector search. 
        Attempts to use 'user_context' for filtering IF implemented (RAG usually struggles here).
        """
        
        # In a generic RAG, 'user_context' is often ignored unless strict filters are hardcoded.
        # We will simulate a "Smart RAG" that tries to filter by department if present.
        
        where_filter = None
        if user_context and "department" in user_context:
            where_filter = {"department": user_context["department"]}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter # This gives RAG a fighting chance compared to strictly naive RAG
        )
        
        return results
