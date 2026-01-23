
import sys

# Mocks
class MockDish:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

class MockResult:
    def __init__(self, top, second=None):
        self.dish_suggestions = [top]
        if second:
            self.dish_suggestions.append(second)
    
    def _is_same_family(self, dish_a, dish_b):
        family_keywords = ["dal", "paneer", "chicken", "biryani", "roti", "paratha", "naan", "curry", "dosa", "idli", "pizza"]
        name_a, name_b = dish_a.lower(), dish_b.lower()
        return any(kw in name_a and kw in name_b for kw in family_keywords)

    def _is_nutrition_distinct(self, dish_a, dish_b, threshold=0.07):
        # Mock nutrition check: "Butter" vs non-butter has high diff
        if "Butter" in dish_a or "Butter" in dish_b: return True
        return False

    def get_best_dish(self, safety_check_fn=None):
        if not self.dish_suggestions: return None
        top = self.dish_suggestions[0]
        if top.confidence < 0.25: return None
        
        # Rule: Existence in DB check (mocked as always True for this test)
        
        if len(self.dish_suggestions) > 1:
            second = self.dish_suggestions[1]
            gap = top.confidence - second.confidence
            if gap < 0.06:
                if self._is_same_family(top.name, second.name):
                    return top.name
                if not self._is_nutrition_distinct(top.name, second.name):
                    return top.name
                if safety_check_fn:
                    if safety_check_fn(top.name) != safety_check_fn(second.name):
                        return None
                return top.name # Top-1 preference
        return top.name

def test_logic():
    print("=== Testing Revised Logic (Isolated) ===")
    
    # Test 1: Permissive Threshold 0.25
    res = MockResult(MockDish("Dal", 0.25))
    assert res.get_best_dish() == "Dal"
    print("Test 1 Passed: Threshold 0.25")

    # Test 2: Same Family resolve
    res = MockResult(MockDish("Dal Makhani", 0.40), MockDish("Dal Fry", 0.38))
    assert res.get_best_dish() == "Dal Makhani"
    print("Test 2 Passed: Same Family")

    # Test 3: Safety mismatch reject
    def safety_bad(name): return "warn" if "Butter" in name else "allow"
    res = MockResult(MockDish("Dal Butter", 0.40), MockDish("Dal Fry", 0.38))
    assert res.get_best_dish(safety_check_fn=safety_bad) is None
    print("Test 3 Passed: Safety Mismatch")

    # Test 4: Top-1 preference (Distinct nutrition but same safety)
    res = MockResult(MockDish("Dal Butter", 0.40), MockDish("Naan", 0.38))
    assert res.get_best_dish(safety_check_fn=lambda x: "allow") == "Dal Butter"
    print("Test 4 Passed: Top-1 Preference")

    print("\n=== LOGIC VERIFIED ===")

if __name__ == "__main__":
    test_logic()
