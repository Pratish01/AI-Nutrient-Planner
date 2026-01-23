
import sys

MAIN_PATH = "src/main.py"
LOG_PATH = "fix_log.txt"

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
                raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")\n"""

def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg)

def run():
    log(f"Reading {MAIN_PATH}...")
    with open(MAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "STABLE PIPELINE" in line:
            log(f"MATCH START at {i}: {line.strip()}")
            # Find the separator line before it
            if i > 0 and "=====" in lines[i-1]:
                start_idx = i - 1
            else:
                start_idx = i
        
        if "Legacy code removed" in line:
            log(f"MATCH END at {i}: {line.strip()}")
            end_idx = i
            
    if start_idx != -1 and end_idx != -1:
        log(f"Replacing lines {start_idx} to {end_idx}")
        # Keep lines up to start_idx (exclusive)
        # Add NEW_LOGIC
        # Keep lines from end_idx+1 (exclusive of end_idx)
        
        # Wait, start_idx is the first line TO REMOVE?
        # Yes.
        
        # New content = lines[:start_idx] + [NEW] + lines[end_idx+1:]
        
        # Check indentation of finally block
        # log(f"Line {end_idx+1}: {lines[end_idx+1]}") # Should be empty
        # log(f"Line {end_idx+2}: {lines[end_idx+2]}") # Should be finally:
        
        final_lines = lines[:start_idx] + [NEW_LOGIC] + lines[end_idx+1:]
        
        try:
            with open(MAIN_PATH, 'w', encoding='utf-8') as f:
                f.writelines(final_lines)
            log("SUCCESS: File written")
        except Exception as e:
            log(f"ERROR writing: {e}")
    else:
        log("FAILED to find blocks")

if __name__ == "__main__":
    # Clear log
    with open(LOG_PATH, "w") as f: f.write("")
    run()
