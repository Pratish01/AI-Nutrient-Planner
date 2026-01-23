
import os

MAIN_PATH = "src/main.py"

# START marker (line 1499)
START_MARKER = "            # ========================================================================="
# END marker (line 1699/1700)
END_MARKER = "            # Legacy code removed. The stable pipeline above returns early upon success or raises an exception."

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
                    # Unknown food
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
                
                # Get nutrition if available (Legacy registry lookup for nutrition data)
                nutrition = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0}
                if nutrition_registry:
                    lookup = nutrition_registry.get_by_name(final_food_name)
                    if lookup:
                        nutrition = lookup
                        print(f"[FOOD UPLOAD] ✓ Nutrition found for {final_food_name}")
                    else:
                        print(f"[FOOD UPLOAD] ! No nutrition data for {final_food_name}")

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

def fix_main():
    with open(MAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_idx = -1
    end_idx = -1
    
    # Simple line scanning because indentation matters
    for i, line in enumerate(lines):
        if START_MARKER.strip() in line:
            # Check if it's the right block (to avoid duplicate markers if any)
            # The legacy block has "STABLE PIPELINE" in the next line usually
            if i+1 < len(lines) and "STABLE PIPELINE" in lines[i+1]:
                start_idx = i
        
        if END_MARKER.strip() in line:
            end_idx = i
            
    if start_idx == -1:
        print("ERROR: Start marker not found!")
        # Fallback search for just the line
        for i, line in enumerate(lines):
             if "STABLE PIPELINE: Food" in line:
                 start_idx = i - 1 # Use the delim line before it
                 break
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        print(f"Replacing block from line {start_idx+1} to {end_idx+1}")
        
        new_lines = lines[:start_idx]
        new_lines.append(NEW_LOGIC + "\n")
        new_lines.extend(lines[end_idx+1:])
        
        with open(MAIN_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        print("File updated successfully.")
    else:
        print(f"Could not find block. Start: {start_idx}, End: {end_idx}")

if __name__ == "__main__":
    fix_main()
