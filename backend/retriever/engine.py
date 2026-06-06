import os
import asyncio
import threading
import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from backend.utils.common import get_logger
from backend.embeddings.embedding_manager import BiomedicalEmbeddingManager

logger = get_logger("CDSS.Retriever")

class VectorDBManager:
    """Thread-safe Singleton Manager for ChromaDB."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorDBManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Resolve path relative to this file
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(BASE_DIR, "vector_db")
        self.collection_name = "rare_diseases"
        self._client = None
        self._collection = None
        # Initialize the high-fidelity biomedical embedding manager
        self._emb_fn = BiomedicalEmbeddingManager("pubmed-bert")
        self._initialized = True

    def get_collection(self):
        """Returns the ChromaDB collection, initializing if necessary."""
        if self._collection is None:
            if not os.path.exists(self.db_path):
                logger.warning(f"Vector DB not found at {self.db_path}")
                return None
            
            try:
                self._client = chromadb.PersistentClient(path=self.db_path)
                # Use our custom biomedical embedding manager as the embedding function
                try:
                    self._collection = self._client.get_collection(
                        name=self.collection_name,
                        embedding_function=self._emb_fn
                    )
                except Exception:
                    # Collection doesn't exist yet — create it with cosine space
                    self._collection = self._client.create_collection(
                        name=self.collection_name,
                        embedding_function=self._emb_fn,
                        metadata={"hnsw:space": "cosine"}
                    )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                return None
        return self._collection

    async def aquery(self, query_texts: List[str], n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Asynchronous wrapper for ChromaDB queries with metadata filtering support."""
        collection = self.get_collection()
        if not collection:
            return {}
            
        loop = asyncio.get_event_loop()
        kwargs = {"query_texts": query_texts, "n_results": n_results}
        if where:
            kwargs["where"] = where
            
        # Run synchronous DB query in executor to preserve async execution loop
        return await loop.run_in_executor(None, lambda: collection.query(**kwargs))


async def retrieve_knowledge_async(symptoms: List[str], filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs async semantic search based on patient symptoms."""
    query = " ".join(symptoms)
    if not query:
        return []

    try:
        manager = VectorDBManager()
        results = await manager.aquery([query], n_results=top_k, where=filters)

        formatted_results = []
        if results and results.get('documents') and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "disease": results['metadatas'][0][i].get('disease', 'Unknown'),
                    "description": results['documents'][0][i],
                    "source": results['metadatas'][0][i].get('source', 'Unknown'),
                    # ChromaDB cosine distance: score = 1 - distance
                    "score": round(max(0.0, 1.0 - results['distances'][0][i]), 2) if 'distances' in results else 0.0
                })
        return formatted_results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

def retrieve_knowledge(symptoms: List[str]) -> List[Dict[str, Any]]:
    """Legacy synchronous semantic search execution."""
    return asyncio.run(retrieve_knowledge_async(symptoms))

def get_knowledge_base_stats() -> dict:
    """Returns exact counts for distinct articles, unique genes, and unique diseases from the vector database."""
    try:
        collection = VectorDBManager().get_collection()
        if not collection:
            return {"articles": 0, "genes": 0, "diseases": 0}
            
        # Get all metadata and IDs to calculate distinct entities
        data = collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])
        
        unique_genes = set()
        unique_diseases = set()
        unique_articles = set()
        
        # Calculate distinct source articles based on document base IDs
        import re
        for doc_id in ids:
            base_id = re.sub(r'_chunk_\d+$', '', doc_id)
            unique_articles.add(base_id)
        
        for meta in metadatas:
            disease = meta.get("disease", "")
            if disease:
                unique_diseases.add(disease)
                
            genes_str = meta.get("genes", "")
            if genes_str:
                for g in genes_str.split(","):
                    g_clean = g.strip()
                    if g_clean:
                        unique_genes.add(g_clean)
                        
        return {
            "articles": len(unique_articles),
            "genes": len(unique_genes),
            "diseases": len(unique_diseases)
        }
    except Exception as e:
        logger.error(f"Error extracting stats: {e}")
        return {"articles": 0, "genes": 0, "diseases": 0}
