"""
Rare Disease CDSS - Biomedical Embedding Pipeline
Uses state-of-the-art biomedical models from HuggingFace for clinical RAG.
Optimized for PubMed, clinical phenotypes, and retrieval-augmented generation.
"""

import torch
import numpy as np
import logging
import hashlib
from typing import List, Union, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from backend.utils.common import get_logger

logger = get_logger("CDSS.Embeddings")

class BiomedicalEmbeddingManager(embedding_functions.EmbeddingFunction):
    """
    Manages loading and execution of real-world biomedical embedding models.
    Supports GPU/CPU fallback, optimal chunking, memory caching, and metadata validation.
    Maintains strict compatibility with ChromaDB's EmbeddingFunction interface.
    """
    
    # Supported Biomedical Models
    MODELS = {
        "pubmed-bert": "NeuML/pubmedbert-base-embeddings",
        "bio-clinical-bert": "emilyalsentzer/Bio_ClinicalBERT",
        "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5"
    }

    # Strict metadata schema required for biomedical evidence indexing
    ALLOWED_METADATA_KEYS = {
        "disease", "gene", "phenotype", "PMID", "source", "confidence"
    }

    def __init__(self, model_key: str = "pubmed-bert"):
        self.model_name = self.MODELS.get(model_key, self.MODELS["pubmed-bert"])
        self._model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache: Dict[str, List[float]] = {}
        logger.info(f"Embedding Manager initialized with {self.model_name} on {self.device}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the embedding model to save GPU/CPU memory."""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully.")
        return self._model

    def _hash_text(self, text: str) -> str:
        """Generates a stable MD5 hash for caching text embeddings."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def optimize_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Splits long clinical or biomedical text into optimized overlapping chunks.
        Approximate token chunking for rapid processing of PubMed abstracts.
        """
        words = text.split()
        chunks = []
        if not words:
            return chunks
            
        for i in range(0, len(words), max(1, chunk_size - overlap)):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures document metadata complies with the strict clinical schema.
        Strips unknown keys silently.
        """
        validated = {}
        for k, v in metadata.items():
            if k in self.ALLOWED_METADATA_KEYS:
                validated[k] = v
            else:
                logger.debug(f"Stripped unsupported metadata key: {k}")
        return validated

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        ChromaDB compatible embedding function with MD5 memory caching.
        Returns a list of embeddings for the given input texts.
        """
        if isinstance(input, str):
            input = [input]
        
        # Ensure strings and strip whitespace
        input_texts = [str(i).strip() if i else " " for i in input]
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []

        # 1. Check Cache
        for idx, text in enumerate(input_texts):
            text_hash = self._hash_text(text)
            if text_hash in self._cache:
                embeddings.append(self._cache[text_hash])
            else:
                embeddings.append(None) # Placeholder to preserve correct index mapping
                texts_to_compute.append(text)
                indices_to_compute.append((idx, text_hash))

        # 2. Compute missing embeddings
        if texts_to_compute:
            try:
                computed = self.model.encode(
                    texts_to_compute, 
                    batch_size=32, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                computed_list = computed.tolist()
                
                # 3. Update Cache & Fill Placeholders
                for i, (original_idx, text_hash) in enumerate(indices_to_compute):
                    self._cache[text_hash] = computed_list[i]
                    embeddings[original_idx] = computed_list[i]
                    
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                raise

        return embeddings

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for a single text string."""
        emb_list = self.__call__([text])[0]
        return np.array(emb_list)

# ─────────────────────────────────────────
# EXECUTABLE USAGE EXAMPLE
# ─────────────────────────────────────────

if __name__ == "__main__":
    manager = BiomedicalEmbeddingManager("pubmed-bert")
    
    clinical_texts = [
        "Patient presents with disproportionate tall stature and aortic root aneurysm.",
        "Differential diagnosis includes Marfan Syndrome and Loeys-Dietz Syndrome."
    ]
    
    print("\n--- BIOMEDICAL EMBEDDING TEST ---")
    print(f"Using Model: {manager.model_name}")
    
    long_text = " ".join(["symptom word"] * 600)
    chunks = manager.optimize_chunking(long_text, chunk_size=512, overlap=50)
    print(f"Chunking test: {len(long_text.split())} words split into {len(chunks)} chunks.")
    
    raw_metadata = {
        "disease": "Marfan",
        "PMID": "12345",
        "irrelevant_key": "drop_me"
    }
    print(f"Validated metadata: {manager.validate_metadata(raw_metadata)}")
    
    embeddings1 = manager(clinical_texts)
    print(f"Generated {len(embeddings1)} embeddings (First run).")
    
    embeddings2 = manager(clinical_texts)
    print(f"Generated {len(embeddings2)} embeddings (Cached run).")
    print(f"Embedding dimension: {len(embeddings1[0])}")
