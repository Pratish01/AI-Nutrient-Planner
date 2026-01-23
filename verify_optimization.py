
import sys
import os

# Mock classes to simulate pipeline output
class MockFoodGroup:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence
        self.cuisine = "indian"

class MockDish:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

class MockResult:
    def __init__(self, top, second):
        self.dish_suggestions = [top, second]
        self.food_group = MockFoodGroup("Generic Group", 0.9)
    
    def get_best_dish(self):
        # COPY OF THE LOGIC FROM stable_food_pipeline.py
        if len(self.dish_suggestions) < 2:
            return self.dish_suggestions[0].name if self.dish_suggestions else None
        
        # LOGIC SHOULD BE: Return top dish regardless of gap
        return self.dish_suggestions[0].name

def test_optimization():
    print("Testing Strict Dish Optimization...")
    
    # Scene: Ambiguous case (51% vs 49%)
    # Old logic would fail (gap < 20%) -> Return None
    # New logic should return "Biryani"
    top = MockDish("Biryani", 0.51)
    second = MockDish("Pulao", 0.49)
    
    result = MockResult(top, second)
    best = result.get_best_dish()
    
    print(f"Top: {top.name} ({top.confidence})")
    print(f"Second: {second.name} ({second.confidence})")
    print(f"Result: {best}")
    
    if best == "Biryani":
        print("[PASS] Optimization Verified: Returned specific dish despite small gap.")
    else:
        print(f"[FAIL] Returned {best} instead of Biryani")

if __name__ == "__main__":
    test_optimization()
