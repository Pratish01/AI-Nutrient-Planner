"""
Quick test script for the food recognition pipeline.
Tests label parsing, model loading, and basic functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_label_parsing():
    """Test TXT file parsing."""
    print("=" * 50)
    print("TEST 1: Label Parsing")
    print("=" * 50)
    
    from services.food_recognition_openclip import LabelFileParser
    
    parser = LabelFileParser()
    
    # Test cuisine file
    cuisines = parser.parse_file("data/CLIP_Cuisine.txt")
    print(f"Cuisine categories: {list(cuisines.keys())}")
    for cat, prompts in cuisines.items():
        print(f"  {cat}: {len(prompts)} prompts")
    
    # Test food groups file
    groups = parser.parse_file("data/CLIP_Food_Groups.txt")
    print(f"\nFood group categories: {list(groups.keys())}")
    for cat, prompts in groups.items():
        print(f"  {cat}: {len(prompts)} prompts")
    
    print("\n[OK] Label parsing test PASSED")
    return True


def test_openclip_import():
    """Test OpenCLIP import."""
    print("\n" + "=" * 50)
    print("TEST 2: OpenCLIP Import")
    print("=" * 50)
    
    try:
        import open_clip
        print(f"OpenCLIP version: {open_clip.__version__ if hasattr(open_clip, '__version__') else 'unknown'}")
        print("[OK] OpenCLIP import PASSED")
        return True
    except ImportError as e:
        print(f"[FAIL] OpenCLIP not installed: {e}")
        print("  Run: pip install open_clip_torch")
        return False


def test_yolo_import():
    """Test YOLO import."""
    print("\n" + "=" * 50)
    print("TEST 3: YOLO Import")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO import PASSED")
        return True
    except ImportError as e:
        print(f"[FAIL] Ultralytics not installed: {e}")
        print("  Run: pip install ultralytics")
        return False


def test_pipeline_init():
    """Test pipeline initialization."""
    print("\n" + "=" * 50)
    print("TEST 4: Pipeline Initialization")
    print("=" * 50)
    
    try:
        from services.food_pipeline import FoodRecognitionPipeline
        
        print("Creating pipeline (this may take a moment)...")
        pipeline = FoodRecognitionPipeline(
            yolo_imgsz=416,
            max_crops=4,
            enable_dish_classification=False
        )
        
        stats = pipeline.get_stats()
        print(f"Pipeline initialized: {stats['initialized']}")
        
        if pipeline.is_available():
            print("[OK] Pipeline initialization PASSED")
            return True
        else:
            print("[FAIL] Pipeline not available after init")
            return False
            
    except Exception as e:
        print(f"[FAIL] Pipeline init failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Food Recognition Pipeline Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Label parsing (no external deps)
    results.append(("Label Parsing", test_label_parsing()))
    
    # Test 2: OpenCLIP import
    results.append(("OpenCLIP Import", test_openclip_import()))
    
    # Test 3: YOLO import
    results.append(("YOLO Import", test_yolo_import()))
    
    # Test 4: Full pipeline (only if deps available)
    if all(r[1] for r in results):
        results.append(("Pipeline Init", test_pipeline_init()))
    else:
        print("\n[WARN] Skipping pipeline test due to missing dependencies")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL]"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
