
import os

MAIN_PATH = "src/main.py"

# The new logic block (Indented 12 spaces)
NEW_LOGIC = """            # =========================================================================
            # CONTINENTAL RETRIEVAL SYSTEM (CLIP)
            # =========================================================================
            try:
                print(f"[FOOD UPLOAD] Calling CONTINENTAL retrieval pipeline...")
                
                # Run retrieval (Top-5)
                # Note: main_inference handles PIL conversion and logic internally
                retrieval_result = continental_system.main_inference(tmp_path, k=5)
                
                # Check status
                if retrieval_result["status"] != "ok":
                    print(f"[FOOD UPLOAD] Retrieval Unsure: {retrieval_result.get('message')}")
                    return {
                        "status": "success",
                        "food_name": "Unknown Food",
                        "confidence": 0.0,
                        "resolution_type": "unknown",
                        "safety_verdict": "unknown",
                        "nutrition": {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0},
                        "meta": {"suggestions": []}
                    }
                
                # Start with best match
                top_match = retrieval_result["top_k_predictions"][0]
                final_food_name = top_match["dish"]
                final_confidence = top_match["score"]
                
                print(f"[FOOD UPLOAD] ✓ Top Match: {final_food_name} ({final_confidence:.3f})")
                
                # Get nutrition if available
                from services.nutrition_registry import get_nutrition_registry
                nutrition_registry = get_nutrition_registry()
                
                nutrition = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0}
                if nutrition_registry:
                    lookup = nutrition_registry.get_by_name(final_food_name)
                    if lookup:
                        nutrition = lookup
                        print(f"[FOOD UPLOAD] ✓ Nutrition found for {final_food_name}")

                # Build final response
                return {
                    "status": "success",
                    "food_name": final_food_name,
                    "confidence": final_confidence,
                    "resolution_type": "retrieval",
                    "safety_verdict": "safe", 
                    "nutrition": {
                        "calories": float(nutrition.get("calories", 0)),
                        "protein_g": float(nutrition.get("protein_g", 0)),
                        "carbs_g": float(nutrition.get("carbs_g", 0)),
                        "fat_g": float(nutrition.get("fat_g", 0)),
                        "sugar_g": float(nutrition.get("sugar_g", 0)),
                        "fiber_g": float(nutrition.get("fiber_g", 0)),
                        "sodium_mg": float(nutrition.get("sodium_mg", 0)),
                    },
                    "meta": {
                        "top_k": retrieval_result["top_k_predictions"]
                    }
                }
                    
            except Exception as e:
                print(f"[FOOD UPLOAD] ERROR: Retrieval failed: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
"""

def fix():
    print(f"Reading {MAIN_PATH}...")
    with open(MAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Locate boundaries
    confidence_line_idx = -1
    finally_line_idx = -1
    
    for i, line in enumerate(lines):
        if "confidence = 0.5" in line:
            confidence_line_idx = i
        
        # We look for the finally block that is after the confidence line
        if confidence_line_idx != -1 and "finally:" in line and "try:" not in line:
            # indent check: matches indent of the 'try' block? 
            # 'confidence' is indent 12. 'finally' should be indent 8.
            if line.startswith("        finally:"):
                finally_line_idx = i
                break # Found it
    
    if confidence_line_idx != -1 and finally_line_idx != -1:
        print(f"Found block: {confidence_line_idx} -> {finally_line_idx}")
        
        # Stitch
        # Part 1: Up to confidence line (inclusive)
        part1 = lines[:confidence_line_idx+1]
        
        # Part 2: New Logic
        part2 = [NEW_LOGIC + "\n"]
        
        # Part 3: From finally line (inclusive)
        part3 = lines[finally_line_idx:]
        
        new_content = part1 + part2 + part3
        
        with open(MAIN_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_content)
        
        print("SUCCESS: File stitched and updated.")
        
    else:
        print(f"ERROR: Could not find markers. Conf: {confidence_line_idx}, Finally: {finally_line_idx}")
        # Debug print lines around where they should be
        print("Searching around line 1495...")
        for j in range(max(0, 1490), min(len(lines), 1500)):
            print(f"{j}: {lines[j].rstrip()}")

if __name__ == "__main__":
    fix()
