import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from services.llm_service import get_llm_service
    from services.rag_service import get_rag_service
    
    # Mock some ingredients
    ingredients = ["moong dal", "spinach", "turmeric"]
    
    # Mock user profile (Diabetic)
    profile = {
        "conditions": ["Diabetes (Type 2)"],
        "allergens": ["Peanuts"]
    }
    
    print("=== AI Recipe Generation Verification ===")
    
    llm = get_llm_service()
    if not llm.is_available:
        print("❌ LLM Service (Ollama) is NOT available. Test will hit fallback logic.")
    else:
        print("✓ LLM Service found.")
        
    print(f"\nGenerating recipe for: {ingredients}")
    print(f"User Profile: {profile}")
    
    print("\n=== ROBUST EXTRACTION TEST ===")
    test_content = "Sure thing! Here is a healthy recipe:\n```json\n{\"name\": \"Test Dish\", \"ingredients\": [\"A\", \"B\"], \"instructions\": [\"1\", \"2\"]}\n```\nHope you like it!"
    
    def extract_json(content):
        import re
        content = content.strip()
        # 1. Try finding JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        else:
            # 2. Try finding the first { and last }
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                return content[start:end+1].strip()
        return content

    extracted = extract_json(test_content)
    print(f"Input: [Noisy content with markdown]")
    try:
        data = json.loads(extracted)
        print(f"✅ Extracted name: {data.get('name')}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

    test_content_2 = "No blocks here! { \"name\": \"No Blocks\", \"ingredients\": [], \"instructions\": [] } Just plain text."
    extracted_2 = extract_json(test_content_2)
    print(f"\nInput: [Noisy content without blocks]")
    try:
        data_2 = json.loads(extracted_2)
        print(f"✅ Extracted name: {data_2.get('name')}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
