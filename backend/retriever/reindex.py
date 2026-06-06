"""
Rare Disease CDSS - Re-indexing Script
Re-creates the vector database using the new Biomedical Embedding Pipeline.
MUST be run after upgrading the embedding model to avoid dimension mismatch.
"""

import os
import pandas as pd
import logging
import shutil
from backend.retriever.engine import VectorDBManager
from backend.utils.common import get_logger

logger = get_logger("CDSS.Reindexer")

def reindex_database():
    """Wipes the existing vector DB and re-indexes the clinical dataset."""
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "vector_db")
    csv_path = os.path.join(BASE_DIR, "data", "cases_clean.csv")
    
    # 1. Load Data
    if not os.path.exists(csv_path):
        logger.error(f"Source data not found at {csv_path}. Please run ingestion first.")
        return

    logger.info("Loading clinical dataset...")
    df = pd.read_csv(csv_path)
    
    # 2. Wipe existing DB
    if os.path.exists(db_path):
        logger.info(f"Removing old vector database at {db_path}...")
        shutil.rmtree(db_path)
    
    # 3. Initialize Manager (this will load the new PubMedBERT model)
    manager = VectorDBManager()
    
    # Force creation of client and collection
    import chromadb
    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection(
        name=manager.collection_name,
        embedding_function=manager._emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 4. Batch Indexing
    logger.info(f"Starting indexing of {len(df)} diseases...")
    
    # Indexing in batches for performance and memory safety
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        
        collection.add(
            ids=[str(idx) for idx in batch.index],
            documents=batch["text"].tolist(),
            metadatas=[
                {
                    "disease": row["disease"],
                    "source": row["source"]
                } for _, row in batch.iterrows()
            ]
        )
        logger.info(f"  Indexed {i + len(batch)} / {len(df)}...")

    logger.info("Re-indexing complete. Your CDSS is now using PubMedBERT embeddings.")

if __name__ == "__main__":
    reindex_database()
