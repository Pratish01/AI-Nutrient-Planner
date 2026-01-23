"""
Test script for the component-driven food recognition system.
"""

import sys
import os
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_component_detection():
    """Test the component detector and rule engine."""
    
    print("=" * 70)
    print("🧪 COMPONENT-DRIVEN RECOGNITION TEST")
    print("=" * 70)
    
    from services.component_detector import get_component_detector
    from services.component_rules import (
        get_component_rule_engine,
        VisualComponent,
        FoodStructure,
    )
    
    # Test 1: Rule Engine Logic (No Image)
    print("\n" + "-" * 70)
    print("TEST 1: Rule Engine Logic (Mock Components)")
    print("-" * 70)
    
    rule_engine = get_component_rule_engine()
    
    # Simulate Pani Puri detection
    pani_puri_components = {
        VisualComponent.HOLLOW_SHELL,
        VisualComponent.SPICED_LIQUID,
        VisualComponent.SMALL_SNACK_SCALE,
    }
    
    result = rule_engine.derive_dish(
        pani_puri_components,
        FoodStructure.DRY_PLUS_LIQUID
    )
    
    print(f"\nInput Components: {[c.value for c in pani_puri_components]}")
    print(f"Resolved Food: {result.resolved_food}")
    print(f"Resolution Type: {result.resolution_type}")
    print(f"Food Group: {result.food_group}")
    print(f"Confidence: {result.confidence:.2f}")
    
    expected = "Pani Puri"
    status = "✅ PASS" if result.resolved_food == expected else "❌ FAIL"
    print(f"Expected: {expected} | {status}")
    
    # Test 2: Mutual Exclusivity Violation
    print("\n" + "-" * 70)
    print("TEST 2: Mutual Exclusivity Violation")
    print("-" * 70)
    
    invalid_components = {
        VisualComponent.HOLLOW_SHELL,
        VisualComponent.SOLID_FRIED_PATTY,  # INVALID: Both present
    }
    
    result = rule_engine.derive_dish(
        invalid_components,
        FoodStructure.UNKNOWN
    )
    
    print(f"\nInput Components: {[c.value for c in invalid_components]}")
    print(f"Resolved Food: {result.resolved_food}")
    print(f"Resolution Type: {result.resolution_type}")
    
    # Should fallback to group due to violation
    status = "✅ PASS" if result.resolution_type == "group" else "❌ FAIL"
    print(f"Expected: Fallback to group | {status}")
    
    # Test 3: Sabudana Vada (Solid Fried)
    print("\n" + "-" * 70)
    print("TEST 3: Sabudana Vada Detection")
    print("-" * 70)
    
    vada_components = {
        VisualComponent.SOLID_FRIED_PATTY,
        VisualComponent.FRIED_TEXTURE,
    }
    
    result = rule_engine.derive_dish(
        vada_components,
        FoodStructure.DRY_ONLY
    )
    
    print(f"\nInput Components: {[c.value for c in vada_components]}")
    print(f"Resolved Food: {result.resolved_food}")
    print(f"Resolution Type: {result.resolution_type}")
    
    expected = "Sabudana Vada"
    status = "✅ PASS" if result.resolved_food == expected else "❌ FAIL"
    print(f"Expected: {expected} | {status}")
    
    # Test 4: Image-Based Detection (if image provided)
    print("\n" + "-" * 70)
    print("TEST 4: Image-Based Component Detection")
    print("-" * 70)
    
    test_images = list(Path("test_images").glob("*.jpg")) + list(Path("test_images").glob("*.png"))
    
    if not test_images:
        print("\n⚠️  No test images found in test_images/")
        print("Add images to test component detection with real images.")
    else:
        detector = get_component_detector()
        
        for img_path in test_images[:3]:  # Test first 3 images
            print(f"\n📷 Testing: {img_path.name}")
            
            image = Image.open(img_path).convert("RGB")
            
            # Get top components
            top_components = detector.get_top_components(image, top_k=5)
            
            print("Top 5 Components:")
            for comp, conf in top_components:
                marker = "✓" if conf >= detector.threshold else " "
                print(f"  [{marker}] {comp}: {conf:.3f}")
            
            # Full detection
            detection = detector.detect(image)
            
            print(f"\nStructure: {detection.structure.value}")
            print(f"Detected: {[c.value for c in detection.detected]}")
            
            # Derive dish
            result = rule_engine.derive_dish(detection.detected, detection.structure)
            
            print(f"\n→ Resolved: {result.resolved_food}")
            print(f"→ Type: {result.resolution_type}")
            print(f"→ Confidence: {result.confidence:.2f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_component_detection()
