import asyncio
from typing import List
from backend.utils.common import get_logger

logger = get_logger("CDSS.NLP.Temporal")

try:
    import medspacy
    nlp = medspacy.load()
except ImportError:
    nlp = None

async def parse_temporal(text: str) -> List[str]:
    """
    Extracts temporal relationships (e.g., onset, duration).
    """
    await asyncio.sleep(0)
    temporal_events = []
    
    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            if getattr(ent._, "is_historical", False):
                temporal_events.append(f"Historical: {ent.text}")
    
    # Fallback/Additional logic
    text_lower = text.lower()
    if "childhood" in text_lower:
        temporal_events.append("Childhood onset")
    if "years" in text_lower or "chronic" in text_lower:
        temporal_events.append("Chronic duration")
    if "days" in text_lower or "hours" in text_lower or "acute" in text_lower:
        temporal_events.append("Acute onset")
    if "sudden" in text_lower:
        temporal_events.append("Sudden onset")
        
    return list(set(temporal_events))
