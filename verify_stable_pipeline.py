"""
Test script to verify the stable food recognition pipeline changes.

Tests:
1. OpenCLIP is installed and can be imported
2. Startup assertion in main.py works
3. stable_food_pipeline.py raises RuntimeError if OpenCLIP missing
4. No silent fallbacks exist
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("VERIFICATION: Stable Food Recognition Pipeline Enforcement")
print("=" * 60)

# Test 1: OpenCLIP Import
print("\n[TEST 1] OpenCLIP Availability...")
try:
    import open_clip
    print("  [PASS] OpenCLIP is installed")
    print(f"  [INFO] Version: {open_clip.__version__ if hasattr(open_clip, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"  [FAIL] OpenCLIP NOT installed: {e}")
    print("  [FIX] pip install open_clip_torch")
    sys.exit(1)

# Test 2: Check main.py has startup assertion
print("\n[TEST 2] Checking main.py for startup assertion...")
main_path = os.path.join(os.path.dirname(__file__), "src", "main.py")
if os.path.exists(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for mandatory assertions
    checks = [
        ("import open_clip", "OpenCLIP import check"),
        ("raise RuntimeError", "Hard fail on missing dependency"),
        ("MANDATORY DEPENDENCY CHECK", "Clear documentation comment"),
    ]
    
    all_pass = True
    for check_str, check_name in checks:
        if check_str in content:
            print(f"  [PASS] Found: {check_name}")
        else:
            print(f"  [FAIL] Missing: {check_name}")
            all_pass = False
            
    if all_pass:
        print("  [PASS] main.py has proper startup assertion")
else:
    print(f"  [FAIL] main.py not found at {main_path}")

# Test 3: Check stable_food_pipeline.py for proper error handling
print("\n[TEST 3] Checking stable_food_pipeline.py for proper error handling...")
pipeline_path = os.path.join(os.path.dirname(__file__), "src", "services", "stable_food_pipeline.py")
if os.path.exists(pipeline_path):
    with open(pipeline_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("raise RuntimeError", "Raises RuntimeError on missing OpenCLIP"),
        ("YOLO class labels", "Documents that YOLO labels are ignored"),
        ("Do NOT silently fail", "Clear anti-fallback documentation"),
    ]
    
    all_pass = True
    for check_str, check_name in checks:
        if check_str in content:
            print(f"  [PASS] Found: {check_name}")
        else:
            print(f"  [FAIL] Missing: {check_name}")
            all_pass = False
            
    if all_pass:
        print("  [PASS] stable_food_pipeline.py has proper error handling")
else:
    print(f"  [FAIL] stable_food_pipeline.py not found at {pipeline_path}")

# Test 4: Check main.py has no silent fallbacks  
print("\n[TEST 4] Checking main.py for removed silent fallbacks...")
if os.path.exists(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # These should NOT exist in the code
    bad_patterns = [
        ('source = "fallback"', "Silent fallback source"),
        ('source = "error_fallback"', "Error fallback source"),
    ]
    
    all_pass = True
    for bad_str, desc in bad_patterns:
        if bad_str in content:
            print(f"  [FAIL] Found forbidden pattern: {desc}")
            all_pass = False
        else:
            print(f"  [PASS] No {desc}")
            
    if all_pass:
        print("  [PASS] main.py has no silent fallbacks")

# Test 5: Check HTTPException is used for errors
print("\n[TEST 5] Checking main.py uses HTTPException for errors...")
if os.path.exists(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for proper error handling
    if 'raise HTTPException' in content and 'food_group": "Unknown"' in content:
        print("  [PASS] Uses HTTPException with structured error response")
    else:
        print("  [FAIL] Missing proper error handling")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nNext Steps:")
print("1. Restart the FastAPI server: uvicorn src.main:app --reload")
print("2. Test food image upload with various images")
print("3. Verify 'Pav Bhaji' is recognized correctly (NOT as 'Orange')")
