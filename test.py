import asyncio
from backend.nlp.medical_ner import extract_lab_values_regex

sample_text = """
Patient presents to the ER with chest pain. 
Lab results show Troponin is 0.4 ng/mL and BNP 450 pg/mL.
Hepatic panel: AST 120 U/L, ALT 85, Bilirubin 1.2 mg/dL.
Renal function is impaired with Creatinine 1.8 mg/dL and BUN 30.
CBC shows WBC 15.2 and Hemoglobin is 8.2 g/dL. Platelets 150.
Patient's glucose is 105 mg/dL.
"""

def test_lab_extraction():
    print("Testing Lab Extraction Regex...")
    print("-" * 40)
    print("Input Text:")
    print(sample_text.strip())
    print("-" * 40)
    
    results = extract_lab_values_regex(sample_text)
    
    print("Extracted Lab Values:")
    if not results:
        print("No lab values extracted.")
    for res in results:
        print(f" - {res}")

if __name__ == "__main__":
    test_lab_extraction()
