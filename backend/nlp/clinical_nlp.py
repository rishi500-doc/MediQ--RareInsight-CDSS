import asyncio
from typing import Dict, Any, List
from backend.utils.common import get_logger
from backend.nlp.entity_extractor import extract_entities
from backend.nlp.negation_detector import detect_negations
from backend.nlp.symptom_normalizer import normalize_symptoms
from backend.nlp.temporal_parser import parse_temporal
from backend.nlp.family_history_parser import parse_family_history
from backend.nlp.medical_ner import extract_medical_ner

logger = get_logger("CDSS.NLP.ClinicalNLP")

class ClinicalNLPEngine:
    """
    Orchestrates the biomedical NLP pipeline to process free-text clinical notes.
    """
    def __init__(self):
        logger.info("Initializing Clinical NLP Engine...")
        
    async def process_clinical_note(self, text: str) -> Dict[str, Any]:
        """
        Main async entry point for processing free-text clinical notes.
        """
        logger.info("Processing clinical note...")
        
        # Run independent extractions concurrently
        entities_task = asyncio.create_task(extract_entities(text))
        ner_task = asyncio.create_task(extract_medical_ner(text))
        temporal_task = asyncio.create_task(parse_temporal(text))
        family_history_task = asyncio.create_task(parse_family_history(text))
        
        entities, ner, temporal, family_history = await asyncio.gather(
            entities_task, ner_task, temporal_task, family_history_task
        )
        
        # Merge extracted entities (symptoms and diseases fall under the broader symptoms category for HPO mapping)
        all_symptoms = list(set(
            entities.get("symptoms", []) + 
            entities.get("diseases", []) + 
            ner.get("symptoms", []) + 
            ner.get("diseases", [])
        ))
        
        # Process negations and normalize
        negated, affirmed = await detect_negations(text, all_symptoms)
        normalized_symptoms = await normalize_symptoms(affirmed)
        
        # Incorporate temporal events and family history accurately
        # Returning strictly the required output structure
        return {
            "symptoms": normalized_symptoms,
            "negated_symptoms": negated,
            "genes": ner.get("genes", []),
            "family_history": family_history,
            "medications": ner.get("medications", []),
            "lab_findings": ner.get("lab_findings", [])
        }
