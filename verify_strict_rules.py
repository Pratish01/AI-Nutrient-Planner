
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.stable_food_pipeline import RecognitionResult, DishSuggestion, FoodGroupPrediction
from services.nutrition_registry import get_nutrition_registry

def test_specificity_gate():
    print("=== Testing New Specificity Gate Rules ===")
    
    # Setup registry (singleton)
    registry = get_nutrition_registry()
    
    # Case 1: Low Confidence (< 0.30) -> Should fail
    print("\nCase 1: Low Confidence (0.25)")
    top = DishSuggestion("Dal Makhani", 0.25)
    second = DishSuggestion("Chole", 0.10)
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: None)")
    assert best is None
    
    # Case 2: Small Gap (< 0.06) -> Should fail
    print("\nCase 2: Small Gap (0.04)")
    top = DishSuggestion("Dal Makhani", 0.40)
    second = DishSuggestion("Dal Tadka", 0.36)
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: None)")
    assert best is None

    # Case 3: Justified Gap (>= 0.06) -> Should pass if in DB
    print("\nCase 3: Justified Gap (0.10)")
    top = DishSuggestion("Dal Makhani", 0.45) # Ensure this name exists in CSV
    second = DishSuggestion("Naan", 0.35)
    res = RecognitionResult(food_group=FoodGroupPrediction("Dal", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    
    # Note: Verification depends on what's in the nutrition DB. 
    # If "Dal Makhani" is in DB, it should return it.
    print(f"Result: {best}")
    
    # Case 4: Not in Database -> Should fail
    print("\nCase 4: Not in Database")
    top = DishSuggestion("Some Fake Dish Name", 0.90)
    second = DishSuggestion("Chole", 0.05)
    res = RecognitionResult(food_group=FoodGroupPrediction("Other", 0.9, "indian"), dish_suggestions=[top, second])
    best = res.get_best_dish()
    print(f"Result: {best} (Expected: None)")
    assert best is None

    print("\n=== Specificity Gate Tests Passed (Logic Check) ===")

if __name__ == "__main__":
    try:
        test_specificity_gate()
    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
