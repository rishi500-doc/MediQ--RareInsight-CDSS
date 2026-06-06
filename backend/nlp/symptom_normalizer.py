import asyncio
from typing import List
from backend.utils.common import get_logger
from backend.hpo.hpo_mapper import HPOMapper

logger = get_logger("CDSS.NLP.Normalizer")

# Preserve compatibility with existing HPO pipeline
hpo_mapper = HPOMapper()

async def normalize_symptoms(symptoms: List[str]) -> List[str]:
    """
    Normalizes extracted symptoms to standard HPO terminology.
    """
    await asyncio.sleep(0)
    normalized = []
    
    for symptom in symptoms:
        mapped = hpo_mapper.map_to_hpo(symptom)
        if mapped and mapped not in normalized:
            normalized.append(mapped)
        elif symptom not in normalized:
            # Fallback to original if no mapping found
            normalized.append(symptom)
            
    return normalized
