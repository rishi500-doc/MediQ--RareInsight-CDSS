"""
Rare Disease CDSS - Genomic Reasoning Agent
Evaluates genomic variants against ClinVar and correlates with HPO phenotypes.
Prioritizes pathogenic variants to generate precise gene-disease matches.
"""

import asyncio
import logging
from typing import List, Dict, Any
from backend.utils.common import get_logger

logger = get_logger("CDSS.GenomicReasoner")

class GenomicReasoner:
    """
    Analyzes patient genomic variants, fetching evidence from ClinVar,
    correlating findings with HPO phenotypic terms, and prioritizing pathogenesis.
    """
    
    def __init__(self):
        # In a full deployment, this would initialize external ClinVar/Ensembl API clients 
        # or local VCF parsers (e.g., PyVCF).
        logger.info("Genomic Reasoner initialized.")

    async def _fetch_clinvar_evidence(self, variant: str) -> Dict[str, Any]:
        """
        Connects to ClinVar evidence asynchronously.
        Simulated here; replace with `httpx` calls to NCBI E-utilities in production.
        """
        await asyncio.sleep(0.05) # Simulate async network/DB delay
        
        variant_lower = variant.lower()
        
        # Simulated pathogenic evidence base for testing CDSS
        if "fbn1" in variant_lower or "c.3037g>a" in variant_lower:
            return {
                "clinical_significance": "Pathogenic",
                "condition": "Marfan Syndrome",
                "review_status": "Reviewed by expert panel",
                "gene": "FBN1"
            }
        elif "brca1" in variant_lower or "brca2" in variant_lower:
            return {
                "clinical_significance": "Pathogenic",
                "condition": "Hereditary breast and ovarian cancer syndrome",
                "review_status": "Criteria provided, multiple submitters",
                "gene": "BRCA1" if "brca1" in variant_lower else "BRCA2"
            }
        elif "cftr" in variant_lower or "deltaf508" in variant_lower:
             return {
                "clinical_significance": "Pathogenic",
                "condition": "Cystic Fibrosis",
                "review_status": "Reviewed by expert panel",
                "gene": "CFTR"
            }
        elif "ttn" in variant_lower:
             return {
                "clinical_significance": "Pathogenic",
                "condition": "Dilated Cardiomyopathy",
                "review_status": "Criteria provided",
                "gene": "TTN"
            }
        else:
            # Default to VUS (Variant of Uncertain Significance)
            parsed_gene = variant.split(" ")[0] if " " in variant else "Unknown"
            return {
                "clinical_significance": "Uncertain significance",
                "condition": "Not provided",
                "review_status": "No assertion criteria provided",
                "gene": parsed_gene
            }

    def _correlate_phenotypes(self, clinvar_data: Dict[str, Any], hpo_terms: List[str]) -> float:
        """
        Correlates genes/conditions from ClinVar with extracted HPO phenotypes.
        Returns a confidence score between 0.0 and 1.0.
        """
        if clinvar_data.get("clinical_significance") != "Pathogenic":
            return 0.0
            
        condition = clinvar_data.get("condition", "").lower()
        score = 0.0
        
        # Simple string-overlap correlation; in production this would map through 
        # an ontology crosswalk (e.g., OMIM ID to HPO ID).
        for hpo in hpo_terms:
            if hpo.lower() in condition:
                 score += 0.5
                 
        # Base boost if pathogenic
        return min(1.0, score + 0.5)

    def _prioritize_variants(self, variant_results: List[Dict[str, Any]], family_history: bool) -> List[Dict[str, Any]]:
        """
        Sorts and filters variants based on pathogenicity correlation and family history.
        """
        pathogenic = []
        for vr in variant_results:
            sig = vr.get("evidence", {}).get("clinical_significance", "").lower()
            if "pathogenic" in sig:
                # Hereditary boost if family history is positive
                priority = vr.get("phenotype_correlation", 0.0)
                if family_history:
                    priority += 0.2
                    
                vr["priority_score"] = min(1.0, round(priority, 2))
                pathogenic.append(vr)
                
        # Sort by highest priority
        return sorted(pathogenic, key=lambda x: x.get("priority_score", 0.0), reverse=True)

    async def analyze(self, variants: List[str], hpo_terms: List[str], family_history: bool = False) -> Dict[str, Any]:
        """
        Main entry point for genomic reasoning.
        Parses variants, connects ClinVar evidence, correlates phenotypes, and generates matches.
        """
        logger.info(f"Analyzing {len(variants)} variants against {len(hpo_terms)} HPO terms...")
        
        if not variants:
            return {"candidate_genes": [], "pathogenic_variants": [], "gene_disease_matches": []}
        
        # 1. Parse and fetch evidence concurrently
        evidence_tasks = [self._fetch_clinvar_evidence(v) for v in variants]
        clinvar_results = await asyncio.gather(*evidence_tasks)
        
        variant_analysis = []
        candidate_genes = set()
        
        # 2. Correlate with phenotypes
        for variant, clinvar_data in zip(variants, clinvar_results):
            gene = clinvar_data.get("gene")
            if gene and gene != "Unknown":
                candidate_genes.add(gene)
                
            correlation = self._correlate_phenotypes(clinvar_data, hpo_terms)
            
            variant_analysis.append({
                "variant": variant,
                "gene": gene,
                "evidence": clinvar_data,
                "phenotype_correlation": correlation
            })
            
        # 3. Prioritize pathogenic variants
        pathogenic_variants = self._prioritize_variants(variant_analysis, family_history)
        
        # 4. Generate Gene-Disease matches
        matches = []
        for pv in pathogenic_variants:
            matches.append({
                "gene": pv["gene"],
                "disease": pv["evidence"]["condition"],
                "variant": pv["variant"],
                "confidence": pv.get("priority_score", 0.8)
            })
            
        # Format output exactly as requested
        return {
            "candidate_genes": list(candidate_genes),
            "pathogenic_variants": pathogenic_variants,
            "gene_disease_matches": matches
        }

# ─────────────────────────────────────────
# EXECUTABLE USAGE EXAMPLE
# ─────────────────────────────────────────

if __name__ == "__main__":
    import json
    
    async def test():
        reasoner = GenomicReasoner()
        
        patient_variants = ["FBN1 c.3037G>A", "UnknownGene c.123T>C", "CFTR deltaF508"]
        patient_hpos = ["tall stature", "lens dislocation", "aortic aneurysm", "marfan syndrome"]
        family_hx = True
        
        print("\n--- GENOMIC REASONER TEST ---")
        results = await reasoner.analyze(
            variants=patient_variants, 
            hpo_terms=patient_hpos, 
            family_history=family_hx
        )
        
        print(json.dumps(results, indent=2))

    asyncio.run(test())
