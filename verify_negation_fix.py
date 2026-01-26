
import re

# Copy of the current logic in src/ocr/parser.py
KNOWN_CONDITIONS = [
    "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
    "hypertension", "high blood pressure", "hbp", "elevated bp",
    "obesity", "overweight"
]

def extract_conditions_test(text):
    conditions = []
    text_lower = text.lower()
    NEGATIONS = ["no", "none", "negative", "not", "without", "denies", "denied", "absent"]
    
    for condition in KNOWN_CONDITIONS:
        pattern = r'\b' + re.escape(condition) + r'\b'
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            start = match.start()
            prefix = text_lower[max(0, start-40):start]
            
            is_negated = False
            for neg in NEGATIONS:
                if re.search(r'\b' + re.escape(neg) + r'\b', prefix):
                    is_negated = True
                    break
            
            if not is_negated:
                normalized = condition.replace("type 1 diabetes", "Type 1 Diabetes").replace("type 2 diabetes", "Type 2 Diabetes")
                normalized = normalized.replace("diabetic", "Diabetes").replace("diabetes", "Diabetes")
                normalized = normalized.replace("hypertension", "Hypertension").replace("high blood pressure", "Hypertension")
                
                cond_title = normalized.title()
                if cond_title not in conditions:
                    conditions.append(cond_title)
                break
    return conditions

def run_tests():
    test_cases = [
        ("Patient has no history of diabetes.", []),
        ("BP is normal, no hypertension.", []),
        ("Patient is diabetic.", ["Diabetes"]),
        ("History of high blood pressure.", ["Hypertension"]),
        ("Denies any history of diabetes or hbp.", []),
        ("Patient is obese and has diabetes.", ["Obesity", "Diabetes"]),
        ("Results for diabetes: negative.", []),
        ("Without any hypertension symptoms.", []),
    ]
    
    passed = 0
    for text, expected in test_cases:
        result = extract_conditions_test(text)
        if sorted(result) == sorted(expected):
            print(f"✅ PASS: '{text}' -> {result}")
            passed += 1
        else:
            print(f"❌ FAIL: '{text}' -> Result: {result}, Expected: {expected}")
            
    print(f"\nPassed {passed}/{len(test_cases)} tests.")

if __name__ == "__main__":
    run_tests()
