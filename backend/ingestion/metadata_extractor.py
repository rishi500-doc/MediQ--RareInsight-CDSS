"""
Rare Disease CDSS - Metadata Extractor
Extracts structured metadata from unstructured biomedical chunks.
"""

import re
from typing import Dict, Any, List

class MetadataExtractor:
    """
    Analyzes raw text chunks to enrich records with structured ontology tags
    like Genes, HPO Terms, and Disease Names.
    """
    def extract_genes(self, text: str) -> List[str]:
        # Simple extraction for demo; would normally route through medspaCy
        genes = set(re.findall(r'\b[A-Z0-9]{3,7}\b', text))
        return list(genes)[:5]

    def extract_hpo(self, text: str) -> List[str]:
        hpo = set(re.findall(r'\bHP:\d{7}\b', text))
        return list(hpo)
        
    def enrich(self, raw_metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
        enriched = raw_metadata.copy()
        if "genes" not in enriched or not enriched["genes"]:
            enriched["genes"] = self.extract_genes(text)
        if "hpo_terms" not in enriched or not enriched["hpo_terms"]:
            enriched["hpo_terms"] = self.extract_hpo(text)
        return enriched
