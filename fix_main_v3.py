
import sys

MAIN_PATH = "src/main.py"

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
                raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")"""

def run():
    print(f"Reading {MAIN_PATH}...", flush=True)
    with open(MAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "STABLE PIPELINE" in line:
            print(f"MATCH START at {i}: {line.strip()}", flush=True)
            start_idx = i - 1 # The line before is the separator
            # Verify the separator exists and looks right
            if "======" in lines[start_idx]:
                 print(f"Confirmed separator at {start_idx}", flush=True)
            else:
                 print(f"Warning: Line {start_idx} is: {lines[start_idx].strip()}", flush=True)
                 # Adjust if needed
                 start_idx = i # Just start at the text line if separator missing
        
        if "Legacy code removed" in line:
            print(f"MATCH END at {i}: {line.strip()}", flush=True)
            end_idx = i
            
    if start_idx != -1 and end_idx != -1:
        print(f"Replacing {start_idx} -> {end_idx}", flush=True)
        new_lines = lines[:start_idx] + [NEW_LOGIC + "\n"] + lines[end_idx+1:]
        
        try:
            with open(MAIN_PATH, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print("SUCCESS: File written", flush=True)
        except Exception as e:
            print(f"ERROR writing: {e}", flush=True)
    else:
        print("FAILED to find blocks", flush=True)
        # Dump some lines to debug
        print("Lines 1495-1505:", flush=True)
        for i in range(1495, 1505):
            if i < len(lines):
                print(f"{i}: {lines[i].rstrip()}", flush=True)

if __name__ == "__main__":
    run()
