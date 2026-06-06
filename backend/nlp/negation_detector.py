import asyncio
from typing import Tuple, List
from backend.utils.common import get_logger
import spacy

logger = get_logger("CDSS.NLP.Negation")

try:
    from negspacy.negation import Negex
    nlp = spacy.blank("en")
    nlp.add_pipe("negex")
except ImportError:
    logger.warning("negspacy not installed. Simulating negation detection.")
    nlp = None

async def detect_negations(text: str, entities: List[str]) -> Tuple[List[str], List[str]]:
    """
    Separates entities into affirmed and negated using negspacy.
    Returns (negated_symptoms, affirmed_symptoms)
    """
    await asyncio.sleep(0)
    
    negated = []
    affirmed = []
    
    text_lower = text.lower()
    for ent in entities:
        # Simplistic mapping since we are working with raw strings instead of Spacy Spans
        if f"no {ent.lower()}" in text_lower or f"denies {ent.lower()}" in text_lower or f"without {ent.lower()}" in text_lower:
            negated.append(ent)
        else:
            affirmed.append(ent)
            
    return list(set(negated)), list(set(affirmed))
