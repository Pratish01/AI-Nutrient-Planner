
import os

MAIN_PATH = "src/main.py"
START_MARKER = '            # STABLE PIPELINE: Food Group = Primary, Dish = Suggestions Only'
END_MARKER = '            # Legacy code removed. The stable pipeline above returns early upon success or raises an exception.'

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
                # Note: This is separate from recognition. Recognition is pure visual retrieval now.
                nutrition = {"calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0, "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0}
                if nutrition_registry:
                    # Try exact lookup or fuzzy match
                    # For now exact name match against registry
                    # (In a real app, you'd want to map specific dishes to nutrition keys)
                    # We will reuse the registry if it has the exact name, else return empty nutrition
                    # This preserves the "Visual Match Only" behavior for missing nutrition
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
                    "safety_verdict": "safe", # Default safe for now, can re-enable rules engine if needed
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

def update_main():
    with open(MAIN_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print("ERROR: Could not find markers!")
        print(f"Start found: {start_idx != -1}")
        print(f"End found: {end_idx != -1}")
        return

    # Keep the end marker line? No, replace up to it.
    # The END_MARKER was inside the block I want to remove? 
    # Actually, the replacement chunk logic ends with `raise HTTPException...`.
    # The end marker `Legacy code removed...` was the LAST line of the block.
    # So I want to replace from START_MARKER to END_MARKER + length of END_MARKER.
    
    new_content = content[:start_idx] + NEW_LOGIC + content[end_idx + len(END_MARKER):]
    
    with open(MAIN_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully updated main.py")

if __name__ == "__main__":
    update_main()
