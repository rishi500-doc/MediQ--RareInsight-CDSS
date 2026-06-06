"""
RAG Bridge — Connects Patient Digital Twin state with the existing RAG pipeline.

When a clinician queries a twin, this module:
1. Builds an enriched medical query from the twin's HPO terms, genetics, and labs
2. Runs it through the existing HybridRetriever
3. Returns rare disease candidates relevant to this specific patient's phenotype

This is the key integration point between the Digital Twin and the existing CDSS.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from backend.utils.common import get_logger

logger = get_logger("CDSS.Twin.RAGBridge")


class TwinRAGBridge:
    """
    Bridges Patient Twin data with the existing RAG pipeline.
    Lazy-loads the HybridRetriever to avoid circular imports.
    """

    def __init__(self):
        self._retriever = None

    def _get_retriever(self):
        """Lazy-load the HybridRetriever."""
        if self._retriever is None:
            from backend.retriever.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever()
        return self._retriever

    def build_rag_query(self, twin: dict) -> str:
        """
        Build an optimised RAG query from the twin's clinical profile.

        Uses HPO labels (standardised medical language) rather than
        free-text symptoms for higher-precision retrieval.
        """
        parts: List[str] = []

        # 1. Active HPO labels (most discriminative)
        hpo_terms = twin.get("hpo_terms", [])
        active_labels = [
            h.get("hpo_label") or h.get("label", "")
            for h in hpo_terms
            if h.get("status", "active") == "active"
        ]
        if active_labels:
            parts.append(f"Phenotype: {', '.join(active_labels[:8])}")

        # 2. Pathogenic genetic variants
        variants = twin.get("genetic_variants", [])
        genes = [
            v.get("gene_symbol", "")
            for v in variants
            if v.get("classification") in ["Pathogenic", "Likely_Pathogenic"]
        ]
        if genes:
            parts.append(f"Genes: {', '.join(genes[:5])}")

        # 3. Demographics
        demo = twin.get("demographics") or {}
        age  = demo.get("age_at_creation")
        if age is not None:
            parts.append(f"Age: {age} years")

        gender = demo.get("gender")
        if gender:
            parts.append(f"Gender: {gender}")

        # 4. Domain context
        parts.append("rare genetic disease diagnosis differential")

        return ". ".join(filter(None, parts))

    def build_patient_dict_for_analyze(self, twin: dict) -> dict:
        """
        Convert a twin into the PatientData dict format expected by
        the existing /analyze endpoint's pipeline (routes.py).

        This allows the existing RAG + reranker + LLM flow to be
        called programmatically from within the twin workflow.
        """
        demo     = twin.get("demographics") or {}
        hpo_terms = twin.get("hpo_terms", [])
        variants  = twin.get("genetic_variants", [])

        # Collect active symptom labels
        symptoms = [
            h.get("hpo_label") or h.get("label", "")
            for h in hpo_terms
            if h.get("status", "active") == "active"
        ]

        # Collect pathogenic gene symbols
        genes = [
            v.get("gene_symbol", "")
            for v in variants
            if v.get("classification") in ["Pathogenic", "Likely_Pathogenic"]
        ]

        fam = twin.get("family_history", [])
        has_family_history = bool(fam)

        return {
            "symptoms"       : symptoms,
            "age"            : demo.get("age_at_creation"),
            "gender"         : demo.get("gender"),
            "family_history" : has_family_history,
            "genomic_info"   : ", ".join(genes) if genes else None,
            "clinical_notes" : twin.get("clinical_notes"),
            "genes"          : genes,
        }

    async def query_with_twin_context(
        self,
        twin: dict,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a full RAG retrieval enriched with the twin's clinical context.

        Returns:
            Dict with rag_query, disease_candidates, and similar_patients
        """
        retriever = self._get_retriever()
        patient_dict = self.build_patient_dict_for_analyze(twin)

        # Use the existing async retrieve path
        candidates = await retriever.retrieve_async(patient_dict)

        # Also find phenotypically similar twins
        twin_id = str(twin.get("twin_id", ""))
        similar = []
        if twin_id:
            try:
                from backend.twin.similarity import find_similar_patients
                similar = find_similar_patients(twin_id, n_results=5, min_similarity=0.60)
            except Exception as e:
                logger.warning(f"Similar patient lookup failed: {e}")

        return {
            "rag_query"          : self.build_rag_query(twin),
            "disease_candidates" : candidates[:n_results],
            "similar_patients"   : similar,
            "twin_id"            : twin_id,
        }
