import asyncio
import spacy
from typing import Dict, List
from backend.utils.common import get_logger

logger = get_logger("CDSS.NLP.EntityExtractor")

try:
    import medspacy
    nlp = medspacy.load()
    logger.info("Loaded medspaCy model.")
except ImportError:
    try:
        # Load scispaCy model if medspaCy is unavailable
        nlp = spacy.load("en_core_sci_sm")
        logger.info("Loaded scispaCy model.")
    except OSError:
        logger.warning("en_core_sci_sm not found. Falling back to default English model.")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")

async def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts base entities using medspaCy or scispaCy.
    """
    await asyncio.sleep(0) # Yield control
    doc = nlp(text)
    
    symptoms = []
    diseases = []
    
    for ent in doc.ents:
        # If medspaCy is used, we can leverage target concepts
        if getattr(ent._, "is_family", False):
            continue  # Handled by family_history_parser
            
        label = ent.label_.upper()
        if label in ["DISEASE", "CONDITION"]:
            diseases.append(ent.text)
        else:
            symptoms.append(ent.text)
            
    return {
        "symptoms": list(set(symptoms)),
        "diseases": list(set(diseases))
    }
