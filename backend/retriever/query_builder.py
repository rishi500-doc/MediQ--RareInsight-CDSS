"""
Rare Disease CDSS - Enhanced Query Builder
Constructs high-fidelity clinical queries by merging raw symptoms, genomic context, and standardized HPO terms.
Designed to be token-efficient for dense embeddings.
"""

from typing import List, Dict, Any, Optional

class ClinicalQueryBuilder:
    """
    Transforms patient data into a dense, token-efficient clinical query for semantic retrieval.
    Balances raw patient descriptions with formal medical ontologies and genomic metadata.
    """
    
    @staticmethod
    def build_search_query(patient_data: Dict[str, Any], mapped_hpos: List[Dict[str, Any]]) -> str:
        """
        Combines symptoms, genes, and HPO terms into a structured, token-optimized query string.
        """
        query_parts = []
        
        # Priority 1: Genomic Context (ClinVar relevance)
        genes = patient_data.get("genes", [])
        if genes:
            query_parts.extend([f"Gene: {g}" for g in genes])
            
        # Priority 2: HPO Standardized Terms
        hpo_terms = [m["term"] for m in mapped_hpos]  
        query_parts.extend(hpo_terms)
        
        # Priority 3: Unique raw symptoms not captured by HPO mapper
        # Preserves the clinician's exact observations
        symptoms = patient_data.get("symptoms", [])
        for s in symptoms:
            if s.lower() not in [t.lower() for t in hpo_terms]:
                query_parts.append(s)
                
        # Priority 4: Clinical Hallmarks & Context (Age, Temporal, Family)
        age = patient_data.get("age")
        if age is not None:
            if age < 2: query_parts.append("infantile onset")
            elif age < 18: query_parts.append("pediatric presentation")
            else: query_parts.append("adult onset")
            
        if patient_data.get("family_history"):
            query_parts.append("hereditary")
            
        # Join into a dense, token-efficient semantic string
        return ", ".join(query_parts)
        
    @staticmethod
    def build_metadata_filters(patient_data: Dict[str, Any], allowed_sources: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Constructs metadata filters for ChromaDB based on patient data and desired sources.
        """
        filters = {}
        
        if allowed_sources:
            if len(allowed_sources) == 1:
                filters["source"] = allowed_sources[0]
            else:
                filters["source"] = {"$in": allowed_sources}
                
        return filters if filters else None
