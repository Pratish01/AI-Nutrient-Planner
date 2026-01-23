"""
Test script for 3-stage hierarchical food recognition.
"""

import sys
import os
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_hierarchical_recognition():
    """Test the 3-stage hierarchical classifier."""
    
    print("=" * 70)
    print("🧪 3-STAGE HIERARCHICAL RECOGNITION TEST")
    print("=" * 70)
    
    from services.hierarchical_classifier import (
        get_hierarchical_classifier,
        VisualTrait,
        FoodType,
        TRAIT_TO_FOOD_TYPES,
    )
    
    # Test 1: Trait → Food Type Mapping
    print("\n" + "-" * 70)
    print("TEST 1: Trait → Food Type Mapping")
    print("-" * 70)
    
    for trait in VisualTrait:
        food_types = TRAIT_TO_FOOD_TYPES.get(trait, [])
        if food_types:
            print(f"  {trait.value:20s} → {[ft.value for ft in food_types]}")
    
    print(f"\n✅ Mappings defined for {len(TRAIT_TO_FOOD_TYPES)} traits")
    
    # Test 2: Image-Based Classification
    print("\n" + "-" * 70)
    print("TEST 2: Image-Based Classification")
    print("-" * 70)
    
    test_images = list(Path("test_images").glob("*.jpg")) + list(Path("test_images").glob("*.png"))
    
    if not test_images:
        print("\n⚠️  No test images found in test_images/")
        print("Add images to test the full hierarchical classification.")
        print("\nTesting with sample image from ultralytics...")
        
        sample_path = Path("venv/Lib/site-packages/ultralytics/assets/bus.jpg")
        if sample_path.exists():
            test_images = [sample_path]
    
    if test_images:
        classifier = get_hierarchical_classifier()
        
        for img_path in test_images[:3]:
            print(f"\n📷 Testing: {img_path.name}")
            
            image = Image.open(img_path).convert("RGB")
            result = classifier.classify(image)
            
            print(f"\n  Stage 1 Traits: {result.stage1_traits}")
            print(f"  Stage 2 Type:   {result.stage2_food_type}")
            print(f"  Resolved Food:  {result.resolved_food}")
            print(f"  Resolution:     {result.resolution_stage}")
            print(f"  Confidence:     {result.confidence:.2f}")
    
    # Test 3: Pipeline Integration
    print("\n" + "-" * 70)
    print("TEST 3: Pipeline Integration")
    print("-" * 70)
    
    try:
        from services.stable_food_pipeline import get_stable_pipeline
        
        pipeline = get_stable_pipeline()
        
        if test_images:
            image = Image.open(test_images[0]).convert("RGB")
            result = pipeline.recognize_hierarchical(image)
            
            print(f"\n  Pipeline Result:")
            print(f"  stage1_traits:     {result.get('stage1_traits')}")
            print(f"  stage2_food_type:  {result.get('stage2_food_type')}")
            print(f"  resolved_food:     {result.get('resolved_food')}")
            print(f"  resolution_stage:  {result.get('resolution_stage')}")
            print(f"  confidence:        {result.get('confidence')}")
            
            print("\n✅ Pipeline integration working!")
    except Exception as e:
        print(f"\n❌ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_hierarchical_recognition()
