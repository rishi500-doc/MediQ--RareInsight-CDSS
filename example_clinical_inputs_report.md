# 10 Example Clinical Inputs for ProHealth CDSS

This report contains 10 structured patient profiles that adhere to the system's `PatientData` schema. They are designed to thoroughly test the NLP symptom extraction, hybrid retrieval, and AI reasoning pipelines of the ProHealth Rare Disease CDSS.

---

### 1. Marfan Syndrome Presentation
**Target Match:** Connective Tissue Disorder
* **Clinical Notes**: "24yo male presenting with tall stature and disproportionately long limbs. Echocardiogram reveals aortic root dilation. Optic exam shows ectopia lentis."
* **Symptoms**: `["tall stature", "arachnodactyly", "lens dislocation", "aortic root aneurysm"]`
* **Age**: `24`
* **Gender**: `"male"`
* **Family History**: `true`
* **Genomic Info**: `"FBN1 mutation suspected"`

---

### 2. Loeys-Dietz Syndrome Presentation
**Target Match:** Connective Tissue Disorder
* **Clinical Notes**: "12-year-old female with widely spaced eyes (hypertelorism) and a cleft palate. Noted to have a bifid uvula on exam. Imaging showed arterial tortuosity."
* **Symptoms**: `["bifid uvula", "hypertelorism", "arterial tortuosity", "cleft palate"]`
* **Age**: `12`
* **Gender**: `"female"`
* **Family History**: `false`
* **Genomic Info**: `"TGFBR1 / TGFBR2"`

---

### 3. Ehlers-Danlos Syndrome Presentation
**Target Match:** Connective Tissue Disorder
* **Clinical Notes**: "19yo female complaining of frequent joint dislocations. Examination shows extreme joint hypermobility and skin hyperextensibility. History of easy bruising and tissue fragility."
* **Symptoms**: `["joint hypermobility", "skin hyperextensibility", "tissue fragility", "frequent dislocations"]`
* **Age**: `19`
* **Gender**: `"female"`
* **Family History**: `true`
* **Genomic Info**: `"COL5A1 / COL5A2 variants"`

---

### 4. Cystic Fibrosis Presentation
**Target Match:** Pulmonary-Gastrointestinal
* **Clinical Notes**: "3yo male infant with failure to thrive. Parents report very salty-tasting skin. History of meconium ileus at birth. Currently exhibiting signs of pancreatic insufficiency."
* **Symptoms**: `["salty skin", "meconium ileus", "pancreatic insufficiency", "steatorrhea", "failure to thrive"]`
* **Age**: `3`
* **Gender**: `"male"`
* **Family History**: `false`
* **Genomic Info**: `"CFTR delta F508"`

---

### 5. Huntington's Disease Presentation
**Target Match:** Neurodegenerative
* **Clinical Notes**: "45yo male presenting with involuntary choreatic movements and progressive cognitive decline. Patient's father had similar symptoms starting in his late 40s."
* **Symptoms**: `["chorea", "cognitive decline", "mood swings", "involuntary movements"]`
* **Age**: `45`
* **Gender**: `"male"`
* **Family History**: `true`
* **Genomic Info**: `"HTT gene CAG repeat expansion"`

---

### 6. Duchenne Muscular Dystrophy Presentation
**Target Match:** Neuromuscular
* **Clinical Notes**: "5yo male with delayed motor milestones and frequent falls. Positive Gowers' sign observed. Calf pseudohypertrophy noted by pediatrician."
* **Symptoms**: `["muscle weakness", "Gowers sign", "calf pseudohypertrophy", "delayed motor milestones"]`
* **Age**: `5`
* **Gender**: `"male"`
* **Family History**: `true`
* **Genomic Info**: `"DMD gene mutation"`

---

### 7. Gaucher Disease Presentation
**Target Match:** Lysosomal Storage Disorder
* **Clinical Notes**: "28yo female of Ashkenazi Jewish descent presenting with hepatosplenomegaly and severe bone pain. Blood work shows anemia and thrombocytopenia."
* **Symptoms**: `["hepatosplenomegaly", "bone pain", "anemia", "thrombocytopenia"]`
* **Age**: `28`
* **Gender**: `"female"`
* **Family History**: `false`
* **Genomic Info**: `"GBA gene mutation"`

---

### 8. Neurofibromatosis Type 1 Presentation
**Target Match:** Multisystemic / Phakomatosis
* **Clinical Notes**: "15yo female presenting with multiple café-au-lait macules and axillary freckling. Lisch nodules observed on slit-lamp examination. Two cutaneous neurofibromas noted."
* **Symptoms**: `["cafe-au-lait macules", "axillary freckling", "Lisch nodules", "neurofibromas"]`
* **Age**: `15`
* **Gender**: `"female"`
* **Family History**: `true`
* **Genomic Info**: `"NF1 gene mutation"`

---

### 9. Pompe Disease Presentation
**Target Match:** Glycogen Storage / Neuromuscular
* **Clinical Notes**: "4-month-old infant presenting with severe hypotonia and failure to thrive. Echocardiogram shows massive cardiomegaly. Elevated CK levels."
* **Symptoms**: `["severe hypotonia", "cardiomegaly", "failure to thrive", "elevated CK"]`
* **Age**: `0`
* **Gender**: `"male"`
* **Family History**: `false`
* **Genomic Info**: `"GAA gene mutation"`

---

### 10. Fabry Disease Presentation
**Target Match:** Lysosomal Storage Disorder
* **Clinical Notes**: "35yo male complaining of burning pain in hands and feet (acroparesthesia). Examination reveals angiokeratomas. Lab shows proteinuria indicating early renal involvement."
* **Symptoms**: `["acroparesthesia", "angiokeratomas", "proteinuria", "corneal verticillata"]`
* **Age**: `35`
* **Gender**: `"male"`
* **Family History**: `true`
* **Genomic Info**: `"GLA gene mutation"`

---

### JSON Usage Example
If you are testing via an API client (like cURL or Postman) against the `/api/v1/analyze` endpoint, you can structure the payload as follows:
```json
{
  "clinical_notes": "12-year-old female with widely spaced eyes (hypertelorism) and a cleft palate. Noted to have a bifid uvula on exam. Imaging showed arterial tortuosity.",
  "symptoms": ["bifid uvula", "hypertelorism", "arterial tortuosity", "cleft palate"],
  "age": 12,
  "gender": "female",
  "family_history": false,
  "genomic_info": "TGFBR1 / TGFBR2"
}
```
