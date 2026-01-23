
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.stable_food_pipeline import RecognitionResult, DishSuggestion, FoodGroupPrediction
from services.nutrition_registry import get_nutrition_registry

def test_revised_resolution_policy():
    print("=== Testing REVISED Permissive Resolution Policy ===")
    
    # 1. Mock Safety Check Function
    def mock_safety_safe(dish_name): return "allow"
    def mock_safety_danger(dish_name): 
        if "Butter" in dish_name: return "warn"
        return "allow"

    # Case 1: Threshold 0.25 (Should PASS now, used to fail at 0.30)
    print("\nCase 1: Threshold 0.26 (Permissive)")
    top = DishSuggestion("Dal Makhani", 0.26)
    second = DishSuggestion("Chole", 0.10)
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: Dal Makhani)")
    assert best == "Dal Makhani"
    
    # Case 2: Small Gap, Same Family (Should PASS)
    print("\nCase 2: Small Gap (0.02), Same Family (Dal)")
    top = DishSuggestion("Dal Makhani", 0.40)
    second = DishSuggestion("Dal Tadka", 0.38)
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: Dal Makhani)")
    assert best == "Dal Makhani"

    # Case 3: Small Gap, Similar Nutrition (Should PASS)
    # Note: This depends on real data, but we can assume for logic check
    print("\nCase 3: Small Gap (0.03), Similar Nutrition")
    top = DishSuggestion("Paneer Tikka", 0.45)
    second = DishSuggestion("Paneer Malai Tikka", 0.42)
    res = RecognitionResult(food_group=FoodGroupPrediction("Paneer", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: Paneer Tikka)")
    assert best == "Paneer Tikka"

    # Case 4: Small Gap, DIFFERENT Safety (Should FAIL/Reject)
    print("\nCase 4: Small Gap (0.03), DIFFERENT Safety Verdicts")
    top = DishSuggestion("Dal Butter Fry", 0.40) # safety: warn
    second = DishSuggestion("Dal Fry", 0.37)       # safety: allow
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish(safety_check_fn=mock_safety_danger)
    print(f"Result: {best} (Expected: None)")
    assert best is None

    # Case 5: Large Gap (Should PASS)
    print("\nCase 5: Large Gap (0.20)")
    top = DishSuggestion("Chicken Biryani", 0.60)
    second = DishSuggestion("Roti", 0.40)
    res = RecognitionResult(food_group=FoodGroupPrediction("Rice", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: Chicken Biryani)")
    assert best == "Chicken Biryani"

    print("\n=== All Revised Resolution Tests Passed! ===")

if __name__ == "__main__":
    try:
        test_revised_resolution_policy()
    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
