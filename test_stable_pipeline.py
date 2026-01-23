"""
Test script for the Stable Food Recognition Pipeline.

Tests:
1. Food group prompts are loaded correctly
2. OpenCLIP model loads
3. YOLO detector loads
4. Pipeline initialization works
5. Dish files are loaded correctly
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def test_food_group_prompts():
    """Test that food group prompts are loaded correctly."""
    print("=" * 50)
    print("TEST 1: Food Group Prompts")
    print("=" * 50)
    
    food_groups_path = DATA_DIR / "CLIP_Food_Groups.txt"
    
    if not food_groups_path.exists():
        print(f"[FAIL] File not found: {food_groups_path}")
        return False
    
    prompts = []
    with open(food_groups_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    
    print(f"Loaded {len(prompts)} food group prompts:")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")
    
    if len(prompts) >= 10:
        print(f"\n[OK] Food group prompts loaded successfully")
        return True
    else:
        print(f"\n[FAIL] Expected at least 10 prompts, got {len(prompts)}")
        return False


def test_dish_files():
    """Test that all dish files exist and have content."""
    print("\n" + "=" * 50)
    print("TEST 2: Dish Files")
    print("=" * 50)
    
    dish_files = [
        "dishes_dal.txt",
        "dishes_rice_dish.txt",
        "dishes_indian_bread.txt",
        "dishes_street_food.txt",
        "dishes_south_indian.txt",
        "dishes_continental.txt",
        "dishes_other.txt",
    ]
    
    all_ok = True
    for fname in dish_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            print(f"  [OK] {fname}: {len(lines)} dishes")
        else:
            print(f"  [FAIL] {fname}: NOT FOUND")
            all_ok = False
    
    if all_ok:
        print(f"\n[OK] All dish files found")
    return all_ok


def test_openclip_import():
    """Test OpenCLIP can be imported."""
    print("\n" + "=" * 50)
    print("TEST 3: OpenCLIP Import")
    print("=" * 50)
    
    try:
        import open_clip
        print(f"OpenCLIP version: {open_clip.__version__}")
        print("[OK] OpenCLIP import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] OpenCLIP not installed: {e}")
        print("  Run: pip install open_clip_torch")
        return False


def test_ultralytics_import():
    """Test YOLO can be imported."""
    print("\n" + "=" * 50)
    print("TEST 4: YOLO Import")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO import successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Ultralytics not installed: {e}")
        print("  Run: pip install ultralytics")
        return False


def test_stable_pipeline_init():
    """Test that the stable pipeline can be initialized."""
    print("\n" + "=" * 50)
    print("TEST 5: Stable Pipeline Initialization")
    print("=" * 50)
    
    try:
        from src.services.stable_food_pipeline import StableFoodPipeline
        
        print("Creating pipeline (this will load models)...")
        pipeline = StableFoodPipeline()
        
        print("Checking if available...")
        available = pipeline.is_available()
        
        print(f"Pipeline available: {available}")
        
        if available:
            print("[OK] Stable pipeline initialized successfully")
            return True
        else:
            print("[FAIL] Pipeline not available")
            return False
            
    except Exception as e:
        print(f"[FAIL] Pipeline initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Stable Food Pipeline Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Food group prompts
    results.append(("Food Groups", test_food_group_prompts()))
    
    # Test 2: Dish files
    results.append(("Dish Files", test_dish_files()))
    
    # Test 3: OpenCLIP
    results.append(("OpenCLIP", test_openclip_import()))
    
    # Test 4: YOLO
    results.append(("YOLO", test_ultralytics_import()))
    
    # Test 5: Pipeline (only if dependencies present)
    if results[2][1] and results[3][1]:
        results.append(("Pipeline Init", test_stable_pipeline_init()))
    else:
        print("\n[SKIP] Pipeline test skipped due to missing dependencies")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL]"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
