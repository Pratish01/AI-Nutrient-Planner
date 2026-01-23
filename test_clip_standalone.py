
import sys
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_clip():
    print("=== Testing CLIP Service ===")
    
    # 1. Check Imports
    try:
        import torch
        import transformers
        from transformers import CLIPProcessor, CLIPModel
        print(f"✓ torch: {torch.__version__}")
        print(f"✓ transformers: {transformers.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return

    # 2. Check Service Initialization
    try:
        from services.clip_service import get_clip_classifier
        print("Initializing CLIP service...")
        clip = get_clip_classifier()
        
        if not clip.is_available:
            print("❌ CLIP service reports not available")
            return
        
        print(f"✓ CLIP service initialized")
        print(f"✓ Vocab size: {len(clip._food_labels)}")
    except Exception as e:
        print(f"❌ Service Init Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test Classification with Dummy Image
    try:
        # Create a red dummy image (apple-like)
        img = Image.new('RGB', (224, 224), color = 'red')
        
        print("\nTesting classification on dummy image...")
        result = clip.classify_image(img)
        
        if result.get("success"):
            print("✓ Classification successful")
            print("Predictions:")
            for p in result.get("predictions", []):
                print(f"  - {p['food_name']}: {p['confidence']:.1%}")
        else:
            print(f"❌ Classification failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Test Run Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_clip()
