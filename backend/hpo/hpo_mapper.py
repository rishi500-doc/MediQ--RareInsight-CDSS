import json
import os
import re
import logging
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CDSS.HPOMapper")

class HPOMapper:
    """Handles mapping of clinical symptoms to HPO terms using local ontology data."""
    
    def __init__(self, dict_path: Optional[str] = None):
        if dict_path is None:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(BASE_DIR, "data", "hpo", "hpo_dictionary.json")
            if os.path.exists(data_path):
                dict_path = data_path
            else:
                # Fallback to the bundled local HPO dictionary for backwards compatibility.
                dict_path = os.path.join(BASE_DIR, "hpo", "hpo_dictionary.json")
        
        self.hpo_data = self._load_dictionary(dict_path)
        self.flat_terms = {}
        self.id_to_primary = {}
        
        # Build synonym expansion maps
        for hpo_id, content in self.hpo_data.items():
            primary_name = content.get("name", "")
            self.id_to_primary[hpo_id] = primary_name
            
            # Map primary name
            if primary_name:
                self.flat_terms[primary_name.lower()] = hpo_id
            
            # Map all synonyms
            for syn in content.get("synonyms", []):
                if syn:
                    self.flat_terms[syn.lower()] = hpo_id

    def _load_dictionary(self, path: str) -> Dict[str, Any]:
        """Loads the HPO dictionary from a JSON file."""
        try:
            if not os.path.exists(path):
                logger.error(f"HPO Dictionary not found at {path}")
                return {}
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading HPO dictionary: {e}")
            return {}

    def normalize_text(self, text: str) -> str:
        """Normalizes symptom text for better fuzzy matching."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    def map_symptom(self, query: str, threshold: float = 80.0) -> Optional[Dict[str, Any]]:
        """
        Maps a single structured symptom string to the best HPO term via fuzzy matching.
        Returns a dictionary with confidence scoring.
        """
        normalized_query = self.normalize_text(query)
        if not normalized_query:
            return None

        # Fuzzy phenotype matching across expanded synonyms
        match = process.extractOne(
            normalized_query, 
            self.flat_terms.keys(), 
            scorer=fuzz.WRatio
        )

        if match:
            matched_term, score, _ = match
            if score >= threshold:
                hpo_id = self.flat_terms[matched_term]
                primary_term = self.id_to_primary[hpo_id]
                
                # Format exactly as required by specifications
                return {
                    "term": primary_term,
                    "hpo_id": hpo_id,
                    "confidence": round(score / 100.0, 2),
                    # Preserved for backward compatibility with older retriever logic
                    "hpo_term": primary_term, 
                    "symptom": query          
                }
            else:
                logger.debug(f"Low confidence match for '{query}': {score}%")
        
        return None

    def map_symptoms(self, symptoms: List[str]) -> List[Dict[str, Any]]:
        """
        Multi-phenotype extraction from a list of structured NLP symptoms.
        Returns a list of extracted HPO terms with confidence scores.
        """
        results = []
        for s in symptoms:
            match = self.map_symptom(s)
            if match:
                # To prevent duplicates from synonym convergence
                if not any(res["hpo_id"] == match["hpo_id"] for res in results):
                    results.append(match)
        return results

    def extract_phenotypes(self, text: str) -> List[Dict[str, Any]]:
        """Multi-phenotype extraction from comma-separated strings."""
        words = text.split(',')
        return self.map_symptoms([w.strip() for w in words if w.strip()])
        
    def map_to_hpo(self, query: str) -> Optional[str]:
        """Simplified string return for older NLP pipeline compatibility."""
        res = self.map_symptom(query)
        if res:
            return res["term"]
        return None

# ─────────────────────────────────────────
# TEST EXAMPLES & EXECUTABLE CODE
# ─────────────────────────────────────────

if __name__ == "__main__":
    mapper = HPOMapper()
    
    # Test cases: exact, typo, synonym
    test_inputs = [
        "tall stature",      # Exact match
        "lens disloction",   # Typo
        "increased height",  # Synonym
        "unrelated thing"    # No match
    ]
    
    print("\n--- HPO MAPPING TEST ---")
    results = mapper.map_symptoms(test_inputs)
    
    for res in results:
        print(f"Input: {res['symptom']}")
        print(f"  -> Term: {res['term']} ({res['hpo_id']})")
        print(f"  -> Confidence: {res['confidence']}")
        print("-" * 30)

    # Realistic Patient Input Example
    patient_input = {
        "symptoms": ["tall stature", "lens dislocation"]
    }
    print(f"\nPatient Analysis for: {patient_input['symptoms']}")
    mapped = mapper.map_symptoms(patient_input["symptoms"])
    print(json.dumps(mapped, indent=2))
