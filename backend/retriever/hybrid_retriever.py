"""
Rare Disease CDSS - Async Hybrid Retrieval Engine
Three-source fusion for 100% rare disease coverage:
  1. ChromaDB semantic vector search (local knowledge base)
  2. HPO reverse phenotype-to-disease lookup (13,000+ diseases, free API)
  3. PubMed live retrieval (fallback for any disease not in ChromaDB or HPO)
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Set
from rapidfuzz import fuzz
from backend.retriever.engine import VectorDBManager
from backend.hpo.hpo_mapper import HPOMapper
from backend.retriever.query_builder import ClinicalQueryBuilder
from backend.retriever.disease_api_client import disease_api_client
from backend.retriever.pubmed_retriever import pubmed_retrieve
from backend.utils.common import get_logger

logger = get_logger("CDSS.HybridRetriever")

class HybridRetriever:
    """
    Implements a multi-stage async retrieval strategy:
    1. Query Formulation (Dense + Token-Efficient)
    2. Semantic Search (Biomedical Vectors)
    3. HPO Ontology Overlap (Phenotypes)
    4. Sparse Keyword Matching (Exact Mentions & Genomics)
    5. Clinical Hallmarks Scoring
    6. Clinical Reranking
    """

    def __init__(self):
        self.vector_manager = VectorDBManager()
        self.hpo_mapper = HPOMapper()
        self.query_builder = ClinicalQueryBuilder()

    def _calculate_hpo_score(self, patient_hpo_ids: Set[str], disease_text: str) -> float:
        if not patient_hpo_ids or not disease_text:
            return 0.0
        
        match_count = 0
        disease_text_lower = disease_text.lower()
        
        for hpo_id in patient_hpo_ids:
            term_name = self.hpo_mapper.id_to_primary.get(hpo_id, "").lower()
            if term_name and term_name in disease_text_lower:
                match_count += 1
                
        return round(match_count / len(patient_hpo_ids), 2)

    def _calculate_sparse_score(self, keywords: List[str], disease_text: str) -> float:
        """Evaluates sparse keyword matches (Symptoms + Genes) in text using rapidfuzz."""
        if not keywords or not disease_text:
            return 0.0
        
        scores = []
        for kw in keywords:
            score = fuzz.partial_ratio(kw.lower(), disease_text.lower())
            scores.append(score / 100.0)
            
        return round(sum(scores) / len(scores), 2) if scores else 0.0
        
    def _calculate_hallmark_score(self, patient_data: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Boosts score if clinical hallmarks (like inheritance patterns or genes) match."""
        score = 0.0
        patient_genes = [g.lower() for g in patient_data.get("genes", [])]
        doc_disease = metadata.get("disease", "").lower()
        
        # Example genomic relevance boost
        for g in patient_genes:
            if g in doc_disease:
                score += 0.5  # High boost for direct gene hit in disease name/metadata
                
        # Inheritance / Hallmark relevancy
        if patient_data.get("family_history") and "autosomal" in doc_disease:
             score += 0.2
             
        return min(1.0, score)

    async def retrieve_async(self, patient_data: Dict[str, Any], top_n: int = 5, sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Three-source fusion retrieval:
          Source 1 — ChromaDB   : semantic vector search over local knowledge base
          Source 2 — HPO Direct : phenotype-to-disease reverse lookup (any of 13k+ diseases)
          Source 3 — PubMed     : live NCBI abstract retrieval (fallback for unknown diseases)
        """
        symptoms = patient_data.get("symptoms", [])
        genes    = patient_data.get("genes", [])

        if not symptoms and not genes:
            return []

        # 1. Map symptoms to HPO terms
        mapped_hpos    = self.hpo_mapper.map_symptoms(symptoms)
        patient_hpo_ids: List[str] = [m["hpo_id"]  for m in mapped_hpos]
        patient_hpo_id_set: Set[str] = set(patient_hpo_ids)

        # 2. Build search query
        query_text = self.query_builder.build_search_query(patient_data, mapped_hpos)
        filters    = self.query_builder.build_metadata_filters(patient_data, allowed_sources=sources)
        logger.info(f"Generated Search Query: {query_text} | Filters: {filters}")

        # ── Run all three sources in parallel ────────────────────────────────
        start_time = time.time()
        chroma_task = self.vector_manager.aquery(
            query_texts=[query_text],
            n_results=top_n * 3,
            where=filters
        )
        hpo_task    = disease_api_client.get_diseases_by_hpo_terms(
            patient_hpo_ids, top_n=top_n * 2
        )
        pubmed_task = pubmed_retrieve(
            symptoms=symptoms, genes=genes, max_results=top_n
        )

        chroma_raw, hpo_candidates, pubmed_candidates = await asyncio.gather(
            chroma_task, hpo_task, pubmed_task
        )
        elapsed = time.time() - start_time
        logger.info(f"HybridRetriever parallel retrieval completed in {elapsed:.2f}s")
        # ────────────────────────────────────────────────────────────────────

        # ── Source 1: Score ChromaDB results ────────────────────────────────
        chroma_results: List[Dict[str, Any]] = []
        if chroma_raw and chroma_raw.get("documents") and chroma_raw["documents"]:
            for i in range(len(chroma_raw["documents"][0])):
                doc_text = chroma_raw["documents"][0][i]
                metadata = chroma_raw["metadatas"][0][i]
                distance = chroma_raw["distances"][0][i]

                semantic_score  = round(max(0.0, 1.0 - distance), 2)
                hpo_score       = self._calculate_hpo_score(patient_hpo_id_set, doc_text)
                sparse_score    = self._calculate_sparse_score(symptoms + genes, doc_text)
                hallmark_score  = self._calculate_hallmark_score(patient_data, metadata)
                final_score     = (semantic_score * 0.35) + (hpo_score * 0.35) + (sparse_score * 0.15) + (hallmark_score * 0.15)

                chroma_results.append({
                    "disease":        metadata.get("disease", "Unknown"),
                    "description":    doc_text,
                    "source":         metadata.get("source", "ChromaDB"),
                    "semantic_score": semantic_score,
                    "hpo_score":      hpo_score,
                    "keyword_score":  sparse_score,
                    "hallmark_score": hallmark_score,
                    "final_score":    round(final_score, 3),
                })
        # ────────────────────────────────────────────────────────────────────

        # ── Merge all three sources, deduplicate by disease name ─────────────
        merged: Dict[str, Dict[str, Any]] = {}

        def _key(name: str) -> str:
            return name.lower().strip()[:60]

        # Priority: ChromaDB (has full text) → HPO-Direct → PubMed
        for r in chroma_results:
            k = _key(r["disease"])
            merged[k] = r

        for r in hpo_candidates:
            k = _key(r["disease"])
            if k not in merged:
                merged[k] = r
            else:
                # Boost existing chroma result with HPO overlap score
                merged[k]["hpo_score"]   = max(merged[k]["hpo_score"], r["hpo_score"])
                merged[k]["final_score"] = round(merged[k]["final_score"] + r["hpo_score"] * 0.20, 3)

        for r in pubmed_candidates:
            k = _key(r["disease"])
            if k not in merged:
                merged[k] = r

        all_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
        logger.info(
            f"Fusion: ChromaDB={len(chroma_results)}, HPO-Direct={len(hpo_candidates)}, "
            f"PubMed={len(pubmed_candidates)} → merged={len(all_results)}"
        )
        # ────────────────────────────────────────────────────────────────────

        return all_results[:top_n]

    def retrieve(self, patient_data: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Legacy synchronous interface"""
        return asyncio.run(self.retrieve_async(patient_data, top_n))

# ─────────────────────────────────────────
# EXECUTABLE TEST CASE
# ─────────────────────────────────────────
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    test_patient = {
        "age": 5,
        "symptoms": ["tall stature", "lens dislocation", "aortic aneurysm"],
        "genes": ["FBN1"],
        "family_history": True
    }
    
    print("\n--- ASYNC HYBRID RETRIEVAL TEST ---")
    results = retriever.retrieve(test_patient, top_n=3)
    
    for res in results:
        print(f"Disease: {res['disease']} [Score: {res['final_score']}]")
        print(f"  - Source: {res['source']}")
        print(f"  - Scores -> Semantic: {res['semantic_score']}, HPO: {res['hpo_score']}, Sparse: {res.get('keyword_score')}, Hallmark: {res.get('hallmark_score')}")
        if 'reasoning' in res:
            print(f"  - Reasoning:")
            for r in res['reasoning']:
                print(f"    - {r}")
        print("-" * 30)
