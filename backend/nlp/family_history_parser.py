import asyncio
from typing import List
from backend.utils.common import get_logger

logger = get_logger("CDSS.NLP.FamilyHistory")

async def parse_family_history(text: str) -> List[str]:
    """
    Parses family history mentions (e.g. maternal aunt, father).
    """
    await asyncio.sleep(0)
    history = []
    lower_text = text.lower()
    
    keywords = ["father", "mother", "sister", "brother", "aunt", "uncle", "grandfather", "grandmother", "sibling", "cousin"]
    
    # Naive extraction, can be replaced by medspaCy's sectionizer
    for kw in keywords:
        if kw in lower_text:
            history.append(f"Positive history related to {kw}")
            
    return list(set(history))
