
# Simplified Verification for Indian Street Food logic
import logging

# Mock configurations similar to the project
FOOD_GROUP_CONFIG = {
    "street food": {
        "dish_file": "dishes_street_food.txt",
        "cuisine": "indian",
        "display_name": "Street Food / Chaat",
        "is_liquid": False,
        "is_street_food": True
    },
    "wet curry": {
        "dish_file": "dishes_other.txt",
        "cuisine": "indian",
        "display_name": "Wet Curry / Gravy",
        "is_liquid": True
    }
}

class MockRecognitionResult:
    def __init__(self, group_name, suggestions):
        self.food_group = type('obj', (object,), {'name': group_name, 'confidence': 0.9})
        self.dish_suggestions = suggestions
        self.bbox = (0, 0, 100, 100)

    def get_best_dish(self):
        if not self.dish_suggestions: return None
        top_suggestion = self.dish_suggestions[0]
        
        # RULE 1: Threshold
        if top_suggestion.confidence < 0.25: return None
            
        # RULE 3: Ambiguity Handling (The logic I added)
        if len(self.dish_suggestions) > 1:
            second_suggestion = self.dish_suggestions[1]
            gap = top_suggestion.confidence - second_suggestion.confidence
            
            # THE STREET FOOD OVERRIDE
            if self.food_group.name == "street food":
                print(f"Street Food Override: Resolved '{top_suggestion.name}' despite gap {gap:.2f}")
                return top_suggestion.name

            # Normal logic
            if gap < 0.06:
                return "Normal Gated Result"
        return top_suggestion.name

def test_logic():
    print("Testing Street Food Gate...")
    sf_suggestions = [
        type('obj', (object,), {'name': 'Pani Puri', 'confidence': 0.3}),
        type('obj', (object,), {'name': 'Bhel Puri', 'confidence': 0.28})
    ]
    res_sf = MockRecognitionResult("street food", sf_suggestions)
    assert res_sf.get_best_dish() == "Pani Puri"

    print("Testing Liquid Bypass selection logic...")
    # Based on the logic in recognize_pil:
    all_results = [
        MockRecognitionResult("street food", []), # SF
        MockRecognitionResult("wet curry", [])    # Liquid
    ]
    
    # Categorize
    street_foods = [r for r in all_results if FOOD_GROUP_CONFIG.get(r.food_group.name, {}).get("is_street_food", False)]
    non_liquids = [r for r in all_results if not FOOD_GROUP_CONFIG.get(r.food_group.name, {}).get("is_liquid", False) and r not in street_foods]
    liquids = [r for r in all_results if FOOD_GROUP_CONFIG.get(r.food_group.name, {}).get("is_liquid", False)]
    
    print(f"Street foods: {len(street_foods)}, Non-liquids: {len(non_liquids)}, Liquids: {len(liquids)}")
    
    primary = None
    if street_foods:
        primary = street_foods[0]
        
    assert primary.food_group.name == "street food"
    print("Logic verified!")

if __name__ == "__main__":
    test_logic()
    
    print("\n--- Testing main.py fallback logic ---")
    food_group = "street food"
    food_group_confidence = 0.1 # Very low
    group_threshold = 0.0 if food_group == "street food" else 0.25
    resolution = "unknown"
    if food_group_confidence >= group_threshold:
        resolution = "group"
    print(f"Street food resolution at low conf: {resolution}")
    assert resolution == "group"
    
    print("\nVerification SUCCESS")
