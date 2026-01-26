
from src.ocr.parser import parse_medical_report
import os
import tempfile

def test_extraction(text, expected_conditions):
    # Create a dummy file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
        f.write(text)
        tmp_path = f.name
    
    try:
        # We need to mock the OCR part to just return our text
        # Since parse_medical_report is complex, let's just test the pattern matching part
        # by looking at how it's implemented and doing a similar check here
        
        from src.ocr.parser import KNOWN_CONDITIONS
        
        text_lower = text.lower()
        found = []
        for condition in KNOWN_CONDITIONS:
            if condition in text_lower:
                normalized = condition.replace("type 1 diabetes", "Type 1 Diabetes").replace("type 2 diabetes", "Type 2 Diabetes")
                normalized = normalized.replace("diabetic", "Diabetes").replace("diabetes", "Diabetes")
                normalized = normalized.replace("hypertension", "Hypertension").replace("high blood pressure", "Hypertension")
                if normalized.title() not in found:
                    found.append(normalized.title())
        
        print(f"Text: '{text}'")
        print(f"Extracted: {found}")
        print(f"Expected:  {expected_conditions}")
        print("-" * 20)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

print("--- Testing Regex Extraction ---")
test_extraction("Patient has no history of diabetes.", ["Diabetes"]) # THIS IS A PROBLEM
test_extraction("BP is normal, no hypertension.", ["Hypertension"]) # THIS IS A PROBLEM
test_extraction("Patient is diabetic.", ["Diabetes"])
test_extraction("History of HBP.", ["Hypertension"]) # Wait, is HBP in KNOWN_CONDITIONS?
