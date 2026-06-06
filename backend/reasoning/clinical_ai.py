"""
Rare Disease CDSS - DeepSeek R1 Reasoning Agent
Integrates patient data, HPO phenotypes, Reranked Evidence, and BioMistral 
summaries to generate deep, explainable clinical reasoning chains.
Streams results via SSE in structured Markdown.
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Generator
from backend.utils.common import get_logger

logger = get_logger("CDSS.DeepSeekR1")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ─── Configurable via .env ─────────────────────────────────────────────────
# Change DEEPSEEK_MODEL in .env to swap models without touching code.
# e.g. deepseek/deepseek-r1-distill-llama-70b  (fast)
#      deepseek/deepseek-chat                    (fastest, no CoT)
#      deepseek/deepseek-r1                      (full reasoning, slowest)
DEEPSEEK_MODEL       = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-r1-distill-llama-70b")
REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "600"))
REASONING_TEMPERATURE= float(os.getenv("REASONING_TEMPERATURE", "0.1"))
# ───────────────────────────────────────────────────────────────────────────
OPENROUTER_API_URL   = "https://openrouter.ai/api/v1/chat/completions"

logger.info(f"Reasoning engine configured → model={DEEPSEEK_MODEL}, max_tokens={REASONING_MAX_TOKENS}, temperature={REASONING_TEMPERATURE}")

def stream_clinical_reasoning(
    patient_data: Dict[str, Any],
    hpo_terms: List[str],
    reranked_evidence: List[Dict[str, Any]],
    biomistral_summary: Dict[str, Any],
    patterns_data: Dict[str, Any],
    recommendations: List[str],
    nlp_summary: Dict[str, Any] = None,
    drug_alerts: List[Dict[str, str]] = None
) -> Generator[str, None, None]:
    """Streams DeepSeek R1 clinical reasoning as Server-Sent Events (SSE)."""

    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'type': 'error', 'delta': 'Server configuration error: OPENROUTER_API_KEY is not set.'})}\n\n"
        return

    # Dynamically inject the newly upgraded Reranker evidence payloads
    context_text = "\n".join([
        f"- {res.get('disease', 'Unknown')} [Confidence: {res.get('confidence_score', 0.0)}]\n"
        f"  * Matched Symptoms: {', '.join(res.get('matched_symptoms', []))}\n"
        f"  * Matched HPO: {', '.join(res.get('matched_hpo_terms', []))}\n"
        f"  * Reranking Explanation: {', '.join(res.get('reranking_explanation', []))}\n"
        for res in reranked_evidence
    ])

    # Inject the summarized payload from the BioMistral inference step
    biomistral_context = (
        f"Summary: {biomistral_summary.get('summary', '')}\n"
        f"Clinical Findings: {', '.join(biomistral_summary.get('clinical_findings', []))}\n"
        f"Genes: {', '.join(biomistral_summary.get('genes', []))}\n"
        f"Phenotypes: {', '.join(biomistral_summary.get('phenotypes', []))}\n"
        f"Associations: {', '.join(biomistral_summary.get('disease_associations', []))}"
    )

    prompt = f"""You are DeepSeek R1, an advanced Clinical Decision Support AI specializing in Rare and Genetic Diseases.

PATIENT DATA:
- Symptoms: {', '.join(patient_data.get('symptoms', []))}
- Age: {patient_data.get('age', 'Not specified')}
- Gender: {patient_data.get('gender', 'Not specified')}
- Family History: {'Positive' if patient_data.get('family_history') else 'Negative'}
- Genomic Variants: {', '.join(patient_data.get('genes', []))}

HPO PHENOTYPES:
{', '.join(hpo_terms)}

BIOMISTRAL COMPRESSED LITERATURE:
{biomistral_context}

RERANKED CLINICAL EVIDENCE:
{context_text}

TASK:
Perform a deep, explainable clinical analysis based on the provided inputs. Preserve Markdown formatting for streaming compatibility. Do not include conversational filler. Structure your response EXACTLY with the following headings to allow downstream UI parsing:

### ALIGNMENT
(Briefly evaluate how the patient's symptoms and HPO terms align with the top RAG matches and literature.)

### REASONING
(Provide an explainable clinical reasoning chain. Include genomic reasoning if genes are present, phenotype-based ranking justifications, symptom severity correlation, and inheritance-aware logic.)

### RECOMMENDATIONS
(Provide diagnostic and treatment recommendations.)

### DIFFERENTIAL DIAGNOSIS
(List the top differential diagnoses. Use bullet points.)

### CONFIDENCE SCORES
(Assign a confidence score/percentage to each differential diagnosis based on the Reranker evidence. Explain the severity or phenotype match driving the score.)

### RELATED GENES
(List any related genes identified in the genomic reasoning or BioMistral summary.)

### SUGGESTED TESTS
(List specific next steps, lab tests, genomic panels, or imaging referrals required to confirm the differential diagnosis.)
"""

    # Emit initial structured metadata mapping to the required output formats
    # Normalize confidence scores so the top result scales to 85% on the UI
    raw_scores = [float(res.get("confidence_score") or 0.0) for res in reranked_evidence]
    max_score  = max(raw_scores) if raw_scores else 1.0
    scale      = 0.85 / max_score if max_score > 0 else 1.0

    # Merge patient-supplied genes + BioMistral-extracted genes immediately
    patient_genes    = patient_data.get("genes", [])
    biomistral_genes = biomistral_summary.get("genes", [])
    all_genes        = list(dict.fromkeys(patient_genes + biomistral_genes))

    # Map raw symptoms to HPO terms to send to the UI for digital twin pre-filling
    from backend.hpo.hpo_mapper import HPOMapper
    mapper = HPOMapper()
    raw_symptoms = patient_data.get("symptoms", [])
    mapped_hpos = []
    for sym in raw_symptoms:
        sym_clean = sym.strip().strip('"').strip("'")
        if not sym_clean:
            continue
        m = mapper.map_symptom(sym_clean)
        if m:
            mapped_hpos.append({
                "hpo_id": m["hpo_id"],
                "hpo_label": m["term"]
            })
        else:
            mapped_hpos.append({
                "hpo_id": "HP:0000000",
                "hpo_label": sym_clean
            })

    metadata = {
        "type": "metadata",
        "nlp_extraction": nlp_summary,
        "drug_alerts": drug_alerts or [],
        "mapped_hpos": mapped_hpos,
        "score": patterns_data.get("score", 0),
        "rare_flag": patterns_data.get("rare_flag", False),
        "patterns": patterns_data.get("patterns", []),
        "possible_conditions": [
            {
                "disease":               res.get("disease", "Unknown"),
                "confidence_score":      res.get("confidence_score", 0.0),
                "confidence_normalized": round(float(res.get("confidence_score") or 0.0) * scale, 3),
                "matched_symptoms":      res.get("matched_symptoms", []),
                "source":                res.get("source", ""),
                "evidence_trail":        res.get("reranking_explanation", []),
                "components":            res.get("components", {})
            }
            for res in reranked_evidence
        ],
        "recommendations": recommendations,
        "differential_diagnosis": [res.get("disease") for res in reranked_evidence],
        "confidence_scores":      [res.get("confidence_score") for res in reranked_evidence],
        "related_genes":          all_genes,
        "suggested_tests":        ["Clinical evaluation"]
    }

    yield f"data: {json.dumps(metadata)}\n\n"

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://prohealth-cdss.ai",
            "X-Title": "ProHealth CDSS"
        }

        request_start = time.time()
        logger.info(f"OpenRouter request starting → model={DEEPSEEK_MODEL}, max_tokens={REASONING_MAX_TOKENS}")
        response = requests.post(
            OPENROUTER_API_URL,
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a clinical decision support AI. Provide structured Markdown reasoning."},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "max_tokens": REASONING_MAX_TOKENS,
                "temperature": REASONING_TEMPERATURE
            },
            headers=headers,
            stream=True,
            timeout=120
        )

        response.raise_for_status()
        logger.info(f"OpenRouter connection established in {time.time() - request_start:.2f}s")
        stream_start = time.time()

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk['choices'][0].get('delta', {})
                    # Yield Markdown content streams
                    if 'content' in delta and delta['content']:
                        yield f"data: {json.dumps({'type': 'content', 'delta': delta['content']})}\n\n"
                    # DeepSeek-R1 supports specific reasoning chain output
                    if 'reasoning' in delta and delta['reasoning']:
                        yield f"data: {json.dumps({'type': 'reasoning', 'delta': delta['reasoning']})}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        logger.info(f"OpenRouter stream completed in {time.time() - stream_start:.2f}s")

    except requests.HTTPError as e:
        logger.error(f"OpenRouter HTTP error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'delta': f'Upstream API error: {e.response.status_code}'})}\n\n"
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'delta': f'System Error: {str(e)}'})}\n\n"
