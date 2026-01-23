
import os
import sys
from PIL import Image
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from services.stable_food_pipeline import get_stable_pipeline

def test_multi_food(image_path):
    print(f"\nEvaluating: {image_path}")
    pipeline = get_stable_pipeline()
    
    # Run recognition
    result = pipeline.recognize(image_path)
    
    print(f"--- PRIMARY RESULTS ---")
    print(f"Food Group: {result.food_group.name} ({result.food_group.confidence:.1%})")
    print(f"BBox: {result.bbox}")
    
    top_dish = result.get_best_dish()
    if top_dish:
        print(f"Best Dish: {top_dish}")
    
    if result.sides:
        print(f"\n--- SIDES DETECTED ({len(result.sides)}) ---")
        for i, side in enumerate(result.sides, 1):
            print(f"Side {i}: {side.food_group.name} (BBox: {side.bbox}, Conf: {side.food_group.confidence:.1%})")
    else:
        print("\nNo sides detected.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("Please provide an image path: python verify_multi_food.py <path>")
        sys.exit(1)
        
    test_multi_food(sys.argv[1])
