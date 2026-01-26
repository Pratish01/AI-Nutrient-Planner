
import re

def test_patterns():
    sample_text = """
    REPORT PREVIEW:
    Name: John Doe
    Gender: Male
    DOB: 15/06/1980
    Height: 180 cm
    Weight: 85 kg
    
    Vitals:
    BP: 135/85
    Glucose Fasting: 110.5
    HbA1c: 6.2 %
    Total Cholesterol: 215.4
    Triglycerides: 160.0
    HDL: 45.3
    LDL: 130.1
    
    Medical History:
    - Hypertension: Yes
    - Diabetes: No
    - Nut allergy: Peanut
    - Medication: Metformin 500mg, Atorvastatin 20mg
    
    Patient denies any history of Chronic Kidney Disease.
    Status: Negative for asthma.
    """
    
    text_lower = sample_text.lower()
    
    # 1. Test Negation logic
    KNOWN_CONDITIONS = ["diabetes", "hypertension", "asthma"]
    NEGATIONS = ["no", "none", "negative", "not", "without", "denies", "denied", "absent"]
    
    detected_conds = []
    for condition in KNOWN_CONDITIONS:
        pattern = r'\b' + re.escape(condition) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        for match in matches:
            start, end = match.start(), match.end()
            prefix = text_lower[max(0, start-40):start]
            suffix = text_lower[end:min(len(text_lower), end+20)]
            is_negated = False
            for neg in NEGATIONS:
                neg_pattern = r'\b' + re.escape(neg) + r'\b'
                if re.search(neg_pattern, prefix) or re.search(neg_pattern, suffix):
                    is_negated = True
                    break
            if not is_negated:
                detected_conds.append(condition)
    
    print(f"Detected Conditions: {detected_conds}")
    
    # 2. Test Vitals logic
    vit = {}
    match_bp = re.search(r'bp[^0-9]*(\d{2,3})[/\s]*(\d{2,3})', text_lower)
    if match_bp:
        vit["systolic_bp"] = float(match_bp.group(1))
        vit["diastolic_bp"] = float(match_bp.group(2))
    
    match_hba1c = re.search(r'hba1c[^0-9]*(\d{1,2}\.?\d?)', text_lower)
    if match_hba1c: vit["hba1c"] = float(match_hba1c.group(1))
    
    print(f"Detected Vitals: {vit}")
    
    # 3. Test Biometrics
    bio = {}
    match_weight = re.search(r'weight[^0-9]*(\d{2,3})', text_lower)
    if match_weight: bio["weight_kg"] = float(match_weight.group(1))
    
    print(f"Detected Biometrics: {bio}")
    
    # 4. Test Medications
    meds = []
    common_meds = ["metformin", "atorvastatin"]
    for med in common_meds:
        if re.search(r'\b' + med + r'\b', text_lower):
            meds.append(med.title())
    
    print(f"Detected Medications: {meds}")
    
    assert "hypertension" in detected_conds
    assert "diabetes" not in detected_conds
    assert "asthma" not in detected_conds
    assert vit["systolic_bp"] == 135.0
    assert vit["hba1c"] == 6.2
    assert bio["weight_kg"] == 85.0
    assert "Metformin" in meds
    
    print("\n✅ Internal Regex Verification SUCCESSFUL!")

if __name__ == "__main__":
    test_patterns()
