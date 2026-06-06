"""
Rare Disease CDSS - Biomedical Chunker
Splits biomedical literature into embedding-ready chunks while preserving semantic context.
"""

from typing import List
from backend.utils.common import get_logger

logger = get_logger("CDSS.Chunker")

class BiomedicalChunker:
    """
    Splits long biomedical texts into optimal token-efficient chunks.
    Preserves overlapping boundaries to maintain semantic continuity across splits.
    """
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        if not words: return chunks
        
        # Sliding window approach for fast, semantic-preserving chunks
        for i in range(0, len(words), max(1, self.chunk_size - self.overlap)):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words): break
            
        return chunks
