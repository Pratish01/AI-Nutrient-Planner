
import re

# Copy of the current logic in src/ocr/parser.py
KNOWN_CONDITIONS = ["diabetes", "hypertension"]
NEGATIONS = ["no", "none", "negative", "not", "without", "denies", "denied", "absent"]

def extract_test(text):
    conditions = []
    text_lower = text.lower()
    
    for condition in KNOWN_CONDITIONS:
        pattern = r'\b' + re.escape(condition) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            start = match.start()
            end = match.end()
            prefix = text_lower[max(0, start-40):start]
            suffix = text_lower[end:min(len(text_lower), end+20)]
            
            is_negated = False
            for neg in NEGATIONS:
                neg_pattern = r'\b' + re.escape(neg) + r'\b'
                if re.search(neg_pattern, prefix) or re.search(neg_pattern, suffix):
                    is_negated = True
                    break
            
            if not is_negated:
                cond_title = condition.title()
                if cond_title not in conditions:
                    conditions.append(cond_title)
                break
    
    vitals = {}
    biometrics = {}
    
    # Age
    match_age = re.search(r'age[^0-9]*(\d{1,2})', text_lower)
    if match_age: biometrics["age"] = int(match_age.group(1))
    
    # BP
    match_bp = re.search(r'bp[^0-9]*(\d{2,3})[/\s]*(\d{2,3})', text_lower)
    if match_bp:
        vitals["systolic_bp"] = float(match_bp.group(1))
        vitals["diastolic_bp"] = float(match_bp.group(2))
        
    return conditions, vitals, biometrics

def run_tests():
    test_cases = [
        ("Diabetes: No", []),
        ("Hypertension: Negative", []),
        ("Patient age: 45, Gender: Male", (45)),
        ("Observed BP: 140/90", (140.0, 90.0)),
        ("Without any history of diabetes", []),
    ]
    
    print("--- Testing Enhanced Parser ---")
    for text, expected in test_cases:
        cond, vit, bio = extract_test(text)
        print(f"Text: '{text}'")
        print(f"  Conditions: {cond}")
        print(f"  Vitals: {vit}")
        print(f"  Biometrics: {bio}")
        print("-" * 20)

if __name__ == "__main__":
    run_tests()
