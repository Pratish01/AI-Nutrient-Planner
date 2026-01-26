
import re
from datetime import datetime

# Simulating the new logic in src/ocr/parser.py
def extract_test(text):
    text_lower = text.lower()
    vitals = {}
    biometrics = {}
    
    # Glucose & HbA1c
    match_gl = re.search(r'(?:glucose|fasting glucose|sugar)[^0-9]*(\d{2,3})', text_lower)
    if match_gl: vitals["glucose_level"] = float(match_gl.group(1))
    
    match_hba1c = re.search(r'hba1c[^0-9]*(\d{1,2}\.?\d?)', text_lower)
    if match_hba1c: vitals["hba1c"] = float(match_hba1c.group(1))
    
    # Lipid Profile
    match_ch = re.search(r'cholesterol[^0-9]*(\d{2,3})', text_lower)
    if match_ch: vitals["cholesterol"] = float(match_ch.group(1))
    
    match_tg = re.search(r'triglycerides[^0-9]*(\d{2,4})', text_lower)
    if match_tg: vitals["triglycerides"] = float(match_tg.group(1))
    
    match_hdl = re.search(r'hdl[^0-9]*(\d{2,3})', text_lower)
    if match_hdl: vitals["hdl"] = float(match_hdl.group(1))
    
    match_ldl = re.search(r'ldl[^0-9]*(\d{2,3})', text_lower)
    if match_ldl: vitals["ldl"] = float(match_ldl.group(1))

    # Biometrics
    match_age = re.search(r'\bage\b[^0-9]*(\d{1,2})', text_lower)
    if match_age: 
        biometrics["age"] = int(match_age.group(1))
    else:
        match_dob = re.search(r'(?:dob|birth)[^0-9]*(\d{2}[\./-]\d{2}[\./-]\d{4})', text_lower)
        if match_dob:
            try:
                dob_str = match_dob.group(1).replace('.', '-').replace('/', '-')
                dob = datetime.strptime(dob_str, "%d-%m-%Y")
                age = 2026 - dob.year # Using 2026 as system year
                biometrics["age"] = age
            except Exception as e:
                print(f"DOB Error: {e}")

    if "female" in text_lower: biometrics["gender"] = "female"
    elif "male" in text_lower: biometrics["gender"] = "male"

    return vitals, biometrics

def run_tests():
    # Simulated OCR text from user image
    ocr_text = """
    Patient Name: Diabetes Profile sample report
    Gender: Female
    Date of Birth: 01.01.1973
    Analysis Result Flag Units Reference Range
    Glucose fasting (PHO) 83 mg/dl 70 - 99
    Cholesterol, total (PHO) 221 high mg/dl 100 - 200
    Triglycerides (PHO) 1315 high mg/dl < 150
    HDL Cholesterol, direct (PHO) 22.5 low mg/dl > 50
    LDL Cholesterol, direct (PHO) 36 mg/dl < 100
    """
    
    vit, bio = extract_test(ocr_text)
    
    print("--- Verification Results ---")
    print(f"Gender: {bio.get('gender')} (Expected: female)")
    print(f"Age: {bio.get('age')} (Expected: ~53)")
    print(f"Glucose: {vit.get('glucose_level')} (Expected: 83.0)")
    print(f"Cholesterol: {vit.get('cholesterol')} (Expected: 221.0)")
    print(f"Triglycerides: {vit.get('triglycerides')} (Expected: 1315.0)")
    print(f"HDL: {vit.get('hdl')} (Expected: 22.0 or 22.5)")
    print(f"LDL: {vit.get('ldl')} (Expected: 36.0)")

if __name__ == "__main__":
    run_tests()
