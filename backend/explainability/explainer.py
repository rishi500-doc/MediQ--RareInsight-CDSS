"""
Rare Disease CDSS - Explainability Agent
Synthesizes the diagnostic reasoning process across the entire RAG pipeline into 
a transparent, structured, and clinician-friendly frontend-ready JSON object.
"""

import logging
from typing import Dict, Any, List
from backend.utils.common import get_logger

logger = get_logger("CDSS.Explainability")

class ClinicalExplainer:
    """
    Constructs transparent, step-by-step diagnostic reasoning explanations for the UI.
    """

    def __init__(self):
        logger.info("Clinical Explainer initialized.")

    def _format_matched_symptoms(self, reranked_disease: Dict[str, Any]) -> List[str]:
        return reranked_disease.get("matched_symptoms", [])

    def _format_hpo_evidence(self, reranked_disease: Dict[str, Any]) -> List[str]:
        return reranked_disease.get("matched_hpo_terms", [])
        
    def _format_pubmed_evidence(self, reranked_disease: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts the literature evidence base."""
        return {
            "source": reranked_disease.get("source", "Unknown"),
            "description": reranked_disease.get("description", "No description available."),
            "semantic_score": reranked_disease.get("components", {}).get("semantic", 0.0)
        }

    def _format_genomic_evidence(self, disease_name: str, genomic_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts matching genomic variants and Pathogenicity from the Genomic Reasoner."""
        evidence = []
        matches = genomic_results.get("gene_disease_matches", [])
        for match in matches:
            # Check for conceptual overlap in disease naming
            if match.get("disease", "").lower() in disease_name.lower() or disease_name.lower() in match.get("disease", "").lower():
                evidence.append({
                    "gene": match.get("gene"),
                    "variant": match.get("variant"),
                    "clinical_significance": "Pathogenic", # Matches are pre-filtered pathogenic hits
                    "confidence": match.get("confidence", 0.0)
                })
        return evidence

    def _generate_clinician_summary(self, disease_name: str, confidence: float, reasoning: List[str]) -> str:
        """Translates machine reasoning into a fluent clinician-friendly summary."""
        if not reasoning:
            return f"Suspected diagnosis of {disease_name} based on baseline clinical similarity."
            
        summary = f"The patient profile aligns with {disease_name} with a confidence score of {confidence * 100:.1f}%. "
        summary += "Key indicators include: " + "; ".join(reasoning) + "."
        return summary

    def build_explainability_report(
        self,
        top_candidates: List[Dict[str, Any]],
        genomic_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates the frontend-ready explainability JSON mapping exactly why a diagnosis was ranked.
        """
        logger.info("Generating frontend explainability report...")
        
        report = {
            "metadata": {
                "total_candidates_analyzed": len(top_candidates),
                "genomic_evidence_present": len(genomic_results.get("pathogenic_variants", [])) > 0
            },
            "diagnoses": []
        }

        for idx, candidate in enumerate(top_candidates):
            disease = candidate.get("disease", "Unknown Condition")
            confidence = candidate.get("confidence_score", 0.0)
            
            # Extract distinct evidential components
            symptoms = self._format_matched_symptoms(candidate)
            hpo_terms = self._format_hpo_evidence(candidate)
            pubmed_evidence = self._format_pubmed_evidence(candidate)
            genomic_evidence = self._format_genomic_evidence(disease, genomic_results)
            reasoning_chain = candidate.get("reranking_explanation", [])
            
            clinician_summary = self._generate_clinician_summary(disease, confidence, reasoning_chain)

            # Assemble strictly defined frontend JSON structure
            diagnosis_report = {
                "rank": idx + 1,
                "disease": disease,
                "confidence_score": confidence,
                "clinician_friendly_explanation": clinician_summary,
                "evidence": {
                    "matched_symptoms": symptoms,
                    "hpo_evidence": hpo_terms,
                    "genomic_evidence": genomic_evidence,
                    "literature_evidence": pubmed_evidence
                },
                "algorithmic_reasoning": reasoning_chain,
                "score_components": candidate.get("components", {})
            }
            
            report["diagnoses"].append(diagnosis_report)

        return report

# ─────────────────────────────────────────
# EXECUTABLE USAGE EXAMPLE
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    
    explainer = ClinicalExplainer()
    
    mock_reranked = [
        {
            "disease": "Marfan syndrome",
            "confidence_score": 0.85,
            "matched_symptoms": ["tall stature", "aortic aneurysm"],
            "matched_hpo_terms": ["Aortic aneurysm", "Ectopia lentis"],
            "description": "FBN1 mutations lead to Marfan syndrome, characterized by tall stature and aortic aneurysm.",
            "source": "PubMed",
            "reranking_explanation": [
                "Hallmark detected: 'Ectopia lentis' aligns strongly with Marfan syndrome",
                "Genomic relevance: Patient genes strongly correlate with this condition"
            ],
            "components": {"semantic": 0.8, "hpo_base": 0.9, "genomic_score": 0.5}
        }
    ]
    
    mock_genomics = {
        "candidate_genes": ["FBN1"],
        "pathogenic_variants": [{"gene": "FBN1", "variant": "FBN1 c.3037G>A"}],
        "gene_disease_matches": [
            {"gene": "FBN1", "disease": "Marfan syndrome", "variant": "FBN1 c.3037G>A", "confidence": 0.9}
        ]
    }
    
    print("\n--- EXPLAINABILITY TEST ---")
    report = explainer.build_explainability_report(mock_reranked, mock_genomics)
    print(json.dumps(report, indent=2))
