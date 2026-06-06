"""
Rare Disease CDSS - BioMistral Inference Engine
Acts as the intermediate summarization layer in the clinical reasoning pipeline:
Retriever -> Reranker -> BioMistral -> DeepSeek R1.

Compresses and interprets dense clinical literature via a real OpenRouter API call,
preparing structured contexts for final deep reasoning by DeepSeek R1.
"""

import os
import time
import asyncio
import json
import httpx
from typing import List, Dict, Any
from backend.utils.common import get_logger

logger = get_logger("CDSS.BioMistral")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BIOMISTRAL_MODEL   = os.getenv("BIOMISTRAL_MODEL", "mistralai/mistral-7b-instruct")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

logger.info(f"BioMistral compression model → {BIOMISTRAL_MODEL}")

class BioMistralEngine:
    """
    Handles biomedical evidence summarization using OpenRouter API.
    Uses a configurable, cost-effective model (default: mistral-7b-instruct)
    to extract structured JSON from retrieved clinical literature.
    """

    def __init__(self):
        logger.info(f"BioMistral Engine ready → model={BIOMISTRAL_MODEL}")

    def _build_compression_prompt(self, reranked_evidence: List[Dict[str, Any]]) -> str:
        """
        Builds a highly constrained prompt to force BioMistral to extract structured JSON.
        """
        prompt = (
            "You are an expert clinical NLP extraction engine.\n"
            "Given the following highly-ranked biomedical evidence from medical literature, "
            "extract and compress the findings into the strict JSON format specified.\n\n"
        )
        
        for idx, ev in enumerate(reranked_evidence):
            disease = ev.get("disease", "Unknown")
            desc = ev.get("description", "")
            prompt += f"--- Evidence {idx+1} ({disease}) ---\n{desc}\n\n"
            
        prompt += (
            "Extract the following structured data:\n"
            "1. A synthesized summary of the clinical literature.\n"
            "2. Notable clinical findings.\n"
            "3. Mentioned genes.\n"
            "4. Mentioned phenotypes.\n"
            "5. Disease associations.\n\n"
            "Output strictly valid JSON matching this schema exactly:\n"
            "{\n"
            '  "summary": "...",\n'
            '  "clinical_findings": [],\n'
            '  "genes": [],\n'
            '  "phenotypes": [],\n'
            '  "disease_associations": []\n'
            "}"
        )
        return prompt

    async def _generate_biomistral_output(self, prompt: str) -> str:
        """
        Calls the OpenRouter API with a fast medical NLP model to extract
        structured JSON from the biomedical evidence prompt.
        Falls back to an empty schema-safe response if the API is unavailable.
        """
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set — BioMistral step skipped.")
            return "{}"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://prohealth-cdss.ai",
            "X-Title":       "ProHealth CDSS BioMistral"
        }
        payload = {
            "model":       BIOMISTRAL_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  600,
            "temperature": 0.0   # Deterministic extraction, no creativity needed
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
                r = await client.post(OPENROUTER_URL, headers=headers, json=payload)
                r.raise_for_status()
                result = r.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"BioMistral API responded ({len(content)} chars)")
                return content
        except Exception as e:
            logger.error(f"BioMistral API call failed: {e}")
            return "{}"   # Return safe empty schema — pipeline continues

    def _parse_and_validate(self, raw_output: str) -> Dict[str, Any]:
        """
        Validates the JSON output against the required schema to prevent downstream DeepSeek crashes.
        """
        default_schema = {
            "summary": "Extraction failed or missing.",
            "clinical_findings": [],
            "genes": [],
            "phenotypes": [],
            "disease_associations": []
        }
        
        try:
            # Strip potential markdown blocks formatting
            clean_text = raw_output.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            
            # Enforce schema keys
            for key in default_schema:
                if key not in data:
                    data[key] = default_schema[key]
                    
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"BioMistral JSON parse error: {e}")
            return default_schema

    async def summarize_evidence(self, reranked_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main execution pipeline:
        Compresses top-k biomedical evidence chunks into a unified JSON structure 
        to be injected safely into the DeepSeek R1 reasoning prompt.
        """
        if not reranked_evidence:
            return self._parse_and_validate("{}")
            
        logger.info(f"BioMistral compressing {len(reranked_evidence)} biomedical evidence chunks...")
        start_time = time.time()
        prompt = self._build_compression_prompt(reranked_evidence)
        raw_llm_output = await self._generate_biomistral_output(prompt)
        structured_data = self._parse_and_validate(raw_llm_output)
        elapsed = time.time() - start_time
        logger.info(f"BioMistral summary completed in {elapsed:.2f}s")
        return structured_data

# ─────────────────────────────────────────
# EXECUTABLE USAGE EXAMPLE
# ─────────────────────────────────────────

if __name__ == "__main__":
    async def test_biomistral():
        engine = BioMistralEngine()
        
        mock_reranked_evidence = [
            {
                "disease": "Marfan syndrome",
                "description": "FBN1 mutations lead to Marfan syndrome, characterized by tall stature and aortic aneurysm."
            },
            {
                "disease": "Loeys-Dietz syndrome",
                "description": "TGFBR1/2 mutations cause Loeys-Dietz, sharing aneurysm phenotypes but lacking ectopia lentis."
            }
        ]
        
        print("\n--- BIOMISTRAL COMPRESSION TEST ---")
        result = await engine.summarize_evidence(mock_reranked_evidence)
        print(json.dumps(result, indent=2))

    asyncio.run(test_biomistral())
