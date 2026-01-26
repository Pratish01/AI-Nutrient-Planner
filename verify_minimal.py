import sys
import os
from pathlib import Path
import csv

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from services.nutrition_registry import get_nutrition_registry

    # Initialize registry
    registry = get_nutrition_registry()
    
    # Test cases
    test_foods = ["Aloo Gobi", "Dal Fry", "Paneer Butter Masala", "Chicken Biryani"]
    
    print("=== Standalone Nutrition Lookup Verification ===")
    for food_name in test_foods:
        print(f"\nFood: {food_name}")
        
        # Test Registry (includes fuzzy match and our new variations)
        res = registry.get_by_name(food_name)
        if res:
            print(f"✓ Found: {res['name']}")
            print(f"  Macros: Cal={res['calories']}, Pro={res['protein_g']}g, Carb={res['carbs_g']}g, Fat={res['fat_g']}g")
        else:
            print(f"❌ Not found in NutritionRegistry")

except Exception as e:
    print(f"Error during verification: {e}")
