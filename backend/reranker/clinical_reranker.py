"""
Rare Disease CDSS - Clinical Reranking System
Refines retrieval candidates using clinically-informed scoring logic, 
age-alignment, family history correlation, genomic weighting, and severity.
"""

import re
import asyncio
import logging
from typing import List, Dict, Any
from rapidfuzz import fuzz
from backend.utils.config_loader import ConfigLoader
from backend.utils.common import get_logger
from backend.retriever.disease_api_client import disease_api_client

logger = get_logger("CDSS.Reranker")

class ClinicalReranker:
    """
    Refines diagnostic candidates using a dynamic, configuration-driven scoring engine.
    Provides traceable, phenotype-aware reasoning for each candidate.
    """

    def __init__(self):
        # Load externalized configurations
        self.hallmarks_config = ConfigLoader.get_hallmarks()
        self.synonyms_config = ConfigLoader.get_synonyms()
        logger.info("ClinicalReranker initialized with dynamic configurations.")

    def _normalize_phenotype(self, text: str) -> str:
        """Cleans and normalizes phenotype text for consistent matching."""
        if not text: return ""
        # Lowercase, remove punctuation, and trim whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join(text.split())

    def _calculate_age_alignment(self, patient_age, disease_text: str) -> float:
        """Determines if the patient's age aligns with typical onset mentioned in text."""
        # Guard: age may be None (optional field) or non-numeric
        if patient_age is None:
            return 0.5
        try:
            patient_age = int(patient_age)
        except (ValueError, TypeError):
            return 0.5
        
        disease_lower = (disease_text or "").lower()
        if patient_age < 18:
            if any(term in disease_lower for term in ["congenital", "infancy", "pediatric", "childhood", "early onset", "juvenile"]):
                return 1.0
        else:
            if any(term in disease_lower for term in ["adult", "late onset", "maturity"]):
                return 1.0
        return 0.5

    def _calculate_family_history_boost(self, has_history: bool, disease_text: str) -> float:
        """Boosts score if disease is explicitly mentioned as hereditary/familial."""
        if not has_history: return 0.0
        
        disease_lower = disease_text.lower()
        if any(term in disease_lower for term in ["hereditary", "familial", "autosomal", "genetic", "inherited", "mutation"]):
            return 1.0
        return 0.5

    def _calculate_genomic_weighting(
        self,
        patient_genes: List[str],
        disease_text: str,
        disease_name: str,
        disease_genes: List[str] = None
    ) -> float:
        """Boosts score if patient genes match the disease's known gene associations.
        Priority order:
          1. Patient gene found in live HPO gene list for this disease  (strongest)
          2. Patient gene mentioned by name in the disease description/name text
        """
        if not patient_genes:
            return 0.0

        score = 0.0
        patient_upper = {g.upper() for g in patient_genes}

        # 1. Check against live HPO gene associations (most reliable)
        if disease_genes:
            known_upper = {g.upper() for g in disease_genes}
            matches = patient_upper & known_upper
            score += min(0.5, len(matches) * 0.25)   # 0.25 per matched gene, cap 0.5
            if matches:
                return round(min(0.5, score), 2)

        # 2. Fallback: literal mention in disease text / name
        disease_lower = disease_text.lower() + " " + disease_name.lower()
        for gene in patient_genes:
            if gene.lower() in disease_lower:
                score += 0.3
        return round(min(0.5, score), 2)

    def _calculate_severity_weighting(self, symptoms: List[str], disease_text: str) -> float:
        """Boosts score if severe symptoms align with the disease description."""
        severe_keywords = ["acute", "severe", "life-threatening", "failure", "aneurysm", "malignancy", "fatal", "crisis"]
        score = 0.0
        disease_lower = disease_text.lower()
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            if any(k in symptom_lower for k in severe_keywords):
                if fuzz.partial_ratio(symptom_lower, disease_lower) >= 80:
                    score += 0.15
        return min(0.3, score)

    def _remove_duplicate_reasoning(self, reasoning: List[str]) -> List[str]:
        """Removes duplicates while preserving the original clinical order."""
        seen = set()
        ordered = []
        for r in reasoning:
            if r not in seen:
                ordered.append(r)
                seen.add(r)
        return ordered

    def rerank(self, patient_data: Dict[str, Any], candidates: List[Dict[str, Any]],
               dynamic_hallmarks: Dict[str, List[str]] = None,
               dynamic_gene_map: Dict[str, List[str]] = None) -> List[Dict[str, Any]]:
        """
        Applies fuzzy clinical weighting with synonym expansion, genomic overlaps, and severity tracking.
        dynamic_hallmarks: optional dict of {disease_name -> [phenotype strings]} from live HPO API.
        dynamic_gene_map:  optional dict of {disease_name -> [gene symbols]}  from live HPO API.
        Falls back to static hallmarks.json if not provided or empty for a disease.
        """
        raw_symptoms = patient_data.get("symptoms", [])
        raw_hpo_terms = patient_data.get("hpo_terms", [])
        raw_genes = patient_data.get("genes", [])
        age = patient_data.get("age")
        has_history = patient_data.get("family_history", False)
        
        # Normalize structures
        symptoms = [self._normalize_phenotype(s) for s in raw_symptoms]
        hpo_terms = [self._normalize_phenotype(h) for h in raw_hpo_terms]
        
        reranked = []
        
        for cand in candidates:
            disease = cand.get("disease", "Unknown")
            desc = self._normalize_phenotype(cand.get("description", ""))
            
            # 1. Base Scores from Retriever (coerce None -> 0.0 defensively)
            semantic = float(cand.get("semantic_score") or 0.0)
            hpo_base = float(cand.get("hpo_score") or 0.0)
            keyword = float(cand.get("keyword_score") or 0.0)
            
            # 2. Clinical Adjustments
            age_score     = self._calculate_age_alignment(age, desc)
            fam_score     = self._calculate_family_history_boost(has_history, desc)
            # Use live HPO gene map for the disease if available, else fall back to text search
            disease_genes = (dynamic_gene_map or {}).get(disease, [])
            genomic_score = self._calculate_genomic_weighting(raw_genes, desc, disease, disease_genes)
            severity_score = self._calculate_severity_weighting(raw_symptoms, desc)
            
            # 3. Phenotype-Aware Reasoning, Overlap Extraction & Boosting
            reasoning = []
            hallmark_boost = 0.0
            matched_symptoms = []
            matched_hpo_terms = []
            
            # HPO term direct overlap extraction
            for h in hpo_terms:
                if h and fuzz.partial_ratio(h, desc) >= 80:
                    matched_hpo_terms.append(h)
                    
            # Hallmark Detection: prefer live API phenotypes, fall back to static config
            disease_info  = self.hallmarks_config.get(disease, {})
            static_marks  = disease_info.get("hallmarks", [])
            live_marks    = (dynamic_hallmarks or {}).get(disease, [])
            # Merge: live API data takes priority, static fills gaps
            hallmarks = live_marks if live_marks else static_marks
            
            for h in hallmarks:
                h_norm = self._normalize_phenotype(h)
                for s in symptoms:
                    sim = fuzz.partial_ratio(h_norm, s)
                    if sim >= 85:
                        hallmark_boost += 0.15
                        reasoning.append(f"Hallmark detected: '{h}' aligns strongly with {disease}.")
                        break
            
            hallmark_boost = min(0.3, hallmark_boost)
            
            # Fuzzy Symptom & Synonym Overlap
            for s, raw_s in zip(symptoms, raw_symptoms):
                synonyms = self.synonyms_config.get(s, [])
                match_pool = [s] + synonyms
                
                found_match = False
                for term in match_pool:
                    term_norm = self._normalize_phenotype(term)
                    sim = fuzz.partial_ratio(term_norm, desc)
                    
                    if sim >= 85:
                        reasoning.append(f"Strong clinical overlap: '{term}' matches disease phenotype.")
                        matched_symptoms.append(raw_s)
                        found_match = True
                        break
                    elif sim >= 70:
                        reasoning.append(f"Partial phenotype overlap: '{term}' detected.")
                        matched_symptoms.append(raw_s)
                        found_match = True
                        break
                
                if not found_match and s in desc:
                    reasoning.append(f"Literal match: '{s}' found in clinical description.")
                    matched_symptoms.append(raw_s)

            if has_history and fam_score > 0:
                reasoning.append(f"Hereditary correlation: Family history supports this genetic suspicion.")
                
            if genomic_score > 0:
                reasoning.append(f"Genomic relevance: Patient genes strongly correlate with this condition.")
                
            if severity_score > 0:
                reasoning.append(f"Severity alignment: High-risk symptoms match the disease profile.")

            # 4. Final Weighted Formula
            final_score = (
                (semantic * 0.25) +
                (hpo_base * 0.25) +
                (keyword * 0.10) +
                (age_score * 0.05) +
                (fam_score * 0.05) +
                (genomic_score * 0.15) +
                (severity_score * 0.05) +
                hallmark_boost
            )
            
            final_score = round(min(1.0, final_score), 2)
            if final_score < 0.45:
                reasoning.append("Note: Clinical matches are currently below high-confidence threshold.")

            # Deduplicate extracted matched attributes
            matched_symptoms = list(set(matched_symptoms))
            matched_hpo_terms = list(set(matched_hpo_terms))

            # 5. Final Assembly (Strict Schema Output)
            reranked.append({
                "disease": disease,
                "confidence_score": final_score,
                "matched_symptoms": matched_symptoms,
                "matched_hpo_terms": matched_hpo_terms,
                "reranking_explanation": self._remove_duplicate_reasoning(reasoning)[:5],
                "source": cand.get("source", "Unknown"),
                # Preserved for transparency and debugging
                "components": {
                    "semantic": semantic,
                    "hpo_base": hpo_base,
                    "keyword": keyword,
                    "age_alignment": age_score,
                    "family_history": fam_score,
                    "genomic_score": genomic_score,
                    "severity_score": severity_score,
                    "hallmark_boost": hallmark_boost
                }
            })

        # Safe sort: coerce None confidence_score to 0.0 to prevent NoneType comparison crash
        reranked.sort(key=lambda x: x.get("confidence_score") or 0.0, reverse=True)
        return reranked

    async def rerank_async(self, patient_data: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Async reranking: fetches live HPO phenotypes AND gene associations for ALL
        candidate diseases in parallel before scoring.
        Both calls share the same underlying cache (_fetch_disease_data), so only
        one HTTP request is made per disease regardless of call order.
        """
        if not candidates:
            return []

        disease_names = [c.get("disease", "") for c in candidates]

        # Fetch phenotypes and gene lists in parallel.
        # DiseaseAPIClient._fetch_disease_data populates both caches in one API call,
        # so the second gather hits the cache and costs zero extra HTTP requests.
        phenotype_lists, gene_lists = await asyncio.gather(
            asyncio.gather(*[disease_api_client.get_disease_phenotypes(name) for name in disease_names]),
            asyncio.gather(*[disease_api_client.get_disease_genes(name)      for name in disease_names]),
        )

        dynamic_hallmarks = {name: ph   for name, ph   in zip(disease_names, phenotype_lists)}
        dynamic_gene_map  = {name: genes for name, genes in zip(disease_names, gene_lists)}

        logger.info(
            f"Live HPO data fetched for {len(disease_names)} diseases. "
            f"Phenotype coverage: {sum(1 for v in dynamic_hallmarks.values() if v)}/{len(disease_names)}  "
            f"Gene coverage: {sum(1 for v in dynamic_gene_map.values() if v)}/{len(disease_names)}"
        )

        # Run CPU-bound reranking in thread executor with live data
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, patient_data, candidates, dynamic_hallmarks, dynamic_gene_map
        )

# ─────────────────────────────────────────
# EXECUTABLE USAGE EXAMPLE
# ─────────────────────────────────────────

if __name__ == "__main__":
    reranker = ClinicalReranker()
    
    mock_candidates = [
        {
            "disease": "Marfan syndrome",
            "description": "Heritable connective tissue disorder characterized by tall stature, ectopia lentis, and severe aortic root aneurysm.",
            "semantic_score": 0.85,
            "hpo_score": 0.80,
            "keyword_score": 0.70,
            "source": "Orphanet"
        }
    ]
    
    test_patient = {
        "age": 12,
        "symptoms": ["tall stature", "lens dislocation", "severe aortic aneurysm"],
        "hpo_terms": ["Aortic aneurysm", "Ectopia lentis"],
        "genes": ["FBN1"],
        "family_history": True
    }
    
    print("\n--- CLINICAL RERANKING TEST ---")
    results = reranker.rerank(test_patient, mock_candidates)
    
    for res in results:
        print(f"Disease: {res['disease']} [Confidence: {res['confidence_score']}]")
        print(f"  Matched Symptoms: {res['matched_symptoms']}")
        print(f"  Matched HPO: {res['matched_hpo_terms']}")
        print(f"  Explanation:")
        for r in res['reranking_explanation']:
            print(f"    - {r}")
        print(f"  Scores: {res['components']}")
        print("-" * 30)
