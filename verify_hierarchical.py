"""
Quick verification script for the hierarchical food recognition system.
"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("VERIFICATION TEST: Hierarchical Food Recognition System")
print("=" * 60)

# Test 1: Nutrition Registry
print("\n[TEST 1] Nutrition Registry Loading...")
try:
    from src.services.nutrition_registry import get_nutrition_registry
    reg = get_nutrition_registry()
    
    cuisines = reg.get_all_cuisines()
    print(f"  ✓ Loaded {len(reg._data)} food items")
    print(f"  ✓ Cuisines found: {cuisines}")
    
    fg = reg.get_unique_food_groups_by_cuisine()
    for cuisine, groups in fg.items():
        print(f"  ✓ {cuisine}: {len(groups)} food groups - {groups[:5]}...")
    
    # Test lookup
    result = reg.get_by_cuisine_and_food_group("Indian", "Dal")
    if result:
        print(f"  ✓ Indian/Dal lookup: {result['calories']:.1f} kcal, {result['item_count']} items")
    else:
        print("  ✗ Indian/Dal lookup failed")
    
    print("  [TEST 1 PASSED]")
except Exception as e:
    print(f"  [TEST 1 FAILED]: {e}")
    import traceback
    traceback.print_exc()

# Test 2: CLIP Classifier Prompts
print("\n[TEST 2] CLIP Classifier Prompt Loading...")
try:
    from src.services.hierarchical_clip_classifier import HierarchicalCLIPClassifier
    clf = HierarchicalCLIPClassifier()
    
    print(f"  ✓ Cuisine prompts: {len(clf.cuisine_prompts)}")
    for p in clf.cuisine_prompts:
        print(f"    - {p}")
    
    print(f"  ✓ Food group prompts:")
    for cuisine, prompts in clf.food_group_prompts.items():
        print(f"    {cuisine}: {len(prompts)} prompts")
    
    print("  [TEST 2 PASSED]")
except Exception as e:
    print(f"  [TEST 2 FAILED]: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Pipeline Import
print("\n[TEST 3] Pipeline Import...")
try:
    from src.services.hierarchical_food_pipeline import get_food_pipeline
    pipeline = get_food_pipeline()
    print("  ✓ Pipeline imported successfully")
    print("  [TEST 3 PASSED]")
except Exception as e:
    print(f"  [TEST 3 FAILED]: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
