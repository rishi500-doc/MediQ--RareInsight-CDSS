"""
Rare Disease CDSS - Vector Indexer
Indexes documents into ChromaDB.
"""
import asyncio
from typing import List, Dict, Any
from backend.retriever.engine import VectorDBManager
from backend.utils.common import get_logger

logger = get_logger("CDSS.VectorIndexer")

class VectorIndexer:
    def __init__(self):
        self.db = VectorDBManager()

    async def index_batch(self, documents: List[Dict[str, Any]]):
        if not documents: return
        
        texts = [doc["chunk_text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [doc["document_id"] for doc in documents]
        
        # We ensure metadata matches chroma required schema (no empty lists, strings etc)
        clean_meta = []
        for meta in metadatas:
            cm = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    cm[k] = ",".join(v)
                elif v:
                    cm[k] = str(v)
            clean_meta.append(cm)
            
        collection = self.db.get_collection()
        if not collection:
            logger.error("Could not get ChromaDB collection to index documents.")
            return
            
        logger.info(f"Indexing {len(texts)} chunks into ChromaDB...")
        
        # Add documents to collection
        collection.add(documents=texts, metadatas=clean_meta, ids=ids)
