import re
import asyncio
from typing import Dict, List
from backend.utils.common import get_logger

logger = get_logger("CDSS.NLP.MedicalNER")

try:
    from transformers import pipeline
    ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
    logger.info("Loaded biomedical transformers pipeline.")
except Exception as e:
    logger.warning(f"Transformers pipeline not loaded. Falling back to mock extraction. Reason: {e}")
    ner_pipeline = None

def _is_valid_medical_term(word: str) -> bool:
    """
    Rejects BERT WordPiece subword tokens and noise before they contaminate symptom lists.
    Valid terms: at least 3 chars, no leading ##, not pure punctuation/digits.
    """
    if not word or len(word.strip()) < 3:
        return False
    if word.startswith("##"):          # BERT subword suffix (##urofibromas, ##yo)
        return False
    if re.fullmatch(r'[\W\d]+', word): # Pure punctuation / digits only
        return False
    return True

def extract_lab_values_regex(text: str) -> List[str]:
    """
    Extracts structured lab results using regex for common clinical note formats.
    Matches: [Test] [optional colon/is] [Value] [Unit]
    """
    results = []
    # System-specific lab tests grouped for LOINC-style extraction
    hepatic    = r"ast|alt|bilirubin|alp|albumin"
    renal      = r"creatinine|bun|gfr|urea"
    cardiac    = r"troponin|bnp|ck-mb|d-dimer"
    hematologic= r"hemoglobin|hgb|hematocrit|wbc|rbc|platelets|inr|ptt"
    metabolic  = r"glucose|hba1c|tsh|calcium|sodium|potassium"
    blood_gas  = r"po2|pco2|ph"
    
    tests = f"{hepatic}|{renal}|{cardiac}|{hematologic}|{metabolic}|{blood_gas}"
    units = r"mg/dl|g/dl|u/l|pg/ml|ng/ml|mmol/l|meq/l|%|k/ul|/ul|fl|pg"
    
    # Matches: [Test Name] [optional separator] [Value] [optional Unit]
    pattern = rf"(?i)\b({tests})\b\s*[:=is]*\s*(\d+(?:\.\d+)?)(?:\s+({units})(?:\b|(?=[\s.,;]|$)))?"
    
    for match in re.finditer(pattern, text):
        test_name = match.group(1).upper() if len(match.group(1)) <= 4 else match.group(1).capitalize()
        value = match.group(2)
        unit = match.group(3) or ""
        results.append(f"{test_name}: {value} {unit}".strip())
        
    return results

async def extract_medical_ner(text: str) -> Dict[str, List[str]]:
    """
    Extracts fine-grained medical entities: genes, medications, lab findings.
    Designed to integrate with biomedical transformers.
    """
    await asyncio.sleep(0)
    
    genes = []
    medications = []
    lab_findings = []
    symptoms = []
    diseases = []
    
    # 1. Regex Extraction for precise lab values
    lab_findings.extend(extract_lab_values_regex(text))
    
    if ner_pipeline:
        # Use transformer pipeline
        # Note: Run in thread pool in real async env if blocking
        entities = ner_pipeline(text)
        for ent in entities:
            label = ent["entity_group"].lower()
            word = ent["word"]
            # --- Sanitize: reject BERT subword artifacts and noise ---
            if not _is_valid_medical_term(word):
                logger.debug(f"Skipping noisy NER token: '{word}'")
                continue
            # ---------------------------------------------------------
            if "gene" in label or "protein" in label:
                genes.append(word)
            elif "medication" in label or "drug" in label or "chemical" in label:
                medications.append(word)
            elif ("lab" in label or "test" in label) and len(lab_findings) == 0:
                # Add NER labs only if regex didn't catch specific structured values
                lab_findings.append(word)
            elif "disease" in label:
                diseases.append(word)
            else:
                symptoms.append(word)
    else:
        # Dummy keyword-based extraction simulating biomedical transformers output
        lower_text = text.lower()
        if "cftr" in lower_text: genes.append("CFTR")
        if "brca" in lower_text: genes.append("BRCA1/2")
        if "ttn" in lower_text: genes.append("TTN")
        
        if "ibuprofen" in lower_text: medications.append("Ibuprofen")
        if "albuterol" in lower_text: medications.append("Albuterol")
        if "metformin" in lower_text: medications.append("Metformin")
        if "lisinopril" in lower_text: medications.append("Lisinopril")
        
        if not lab_findings: # Only use generic if no specific lab values found
            if "wbc" in lower_text or "leukocytes" in lower_text: lab_findings.append("Elevated WBC")
            if "hemoglobin" in lower_text or "anemia" in lower_text: lab_findings.append("Low Hemoglobin")
            if "glucose" in lower_text: lab_findings.append("Abnormal Glucose")
    
    return {
        "genes": list(set(genes)),
        "medications": list(set(medications)),
        "lab_findings": list(set(lab_findings)),
        "symptoms": list(set(symptoms)),
        "diseases": list(set(diseases))
    }
