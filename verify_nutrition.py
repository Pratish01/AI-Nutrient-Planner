import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from services.nutrition_registry import get_nutrition_registry
    import main

    # Initialize registry
    registry = get_nutrition_registry()
    
    # Test cases
    test_foods = ["Aloo Gobi", "Dal Fry", "Paneer Butter Masala", "Chicken Biryani"]
    
    print("=== Nutrition Lookup Verification ===")
    for food_name in test_foods:
        print(f"\nFood: {food_name}")
        
        # 1. Test Registry (Fuzzy Match)
        res_registry = registry.get_by_name(food_name)
        if res_registry:
            print(f"[Registry] ✓ Found: {res_registry['name']}")
            print(f"[Registry]   Macros: Cal={res_registry['calories']}, Pro={res_registry['protein_g']}g, Carb={res_registry['carbs_g']}g, Fat={res_registry['fat_g']}g")
        else:
            print(f"[Registry] ❌ Not found")
            
        # 2. Test Local FOOD_DATABASE (Exact Match)
        res_db = main.FOOD_DATABASE.get(food_name.lower())
        if res_db:
            print(f"[FOOD_DATABASE] ✓ Found exact match")
            print(f"[FOOD_DATABASE]   Macros: Cal={res_db['calories']}, Pro={res_db['protein_g']}g, Carb={res_db['carbs_g']}g, Fat={res_db['fat_g']}g")
        else:
            print(f"[FOOD_DATABASE] ❌ No exact match")

except Exception as e:
    print(f"Error during verification: {e}")
    import traceback
    traceback.print_exc()
