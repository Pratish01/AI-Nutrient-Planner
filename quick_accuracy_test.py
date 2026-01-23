"""
Quick accuracy test using a sample image.
This will show you exactly what OpenCLIP is predicting at each stage.
"""

import sys
import os
from pathlib import Path
from PIL import Image
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_with_sample_image():
    """Test with ultralytics sample image (bus.jpg)."""
    
    # Use the ultralytics sample image
    sample_image_path = Path("venv/Lib/site-packages/ultralytics/assets/bus.jpg")
    
    if not sample_image_path.exists():
        print(f"❌ Sample image not found: {sample_image_path}")
        print("\nPlease provide your own test image:")
        print("  python quick_accuracy_test.py path/to/your/food/image.jpg")
        return
    
    print("=" * 70)
    print("🧪 QUICK ACCURACY TEST")
    print("=" * 70)
    print(f"Using sample image: {sample_image_path}")
    
    # Load image
    image = Image.open(sample_image_path).convert("RGB")
    print(f"✓ Image loaded: {image.size}")
    
    # Test 1: Standalone OpenCLIP
    print("\n" + "-" * 70)
    print("TEST 1: Standalone OpenCLIP (Full Image)")
    print("-" * 70)
    
    try:
        import open_clip
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained='laion2b_s34b_b88k'
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Load food group prompts
        prompts_path = Path("data/CLIP_Food_Groups.txt")
        prompts = []
        with open(prompts_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
        
        # Encode
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        
        print("\nTop 5 Food Group Predictions:")
        for i in range(min(5, len(prompts))):
            idx = sorted_idx[i].item()
            prob = sorted_probs[i].item()
            print(f"  {i+1}. [{prob:.3f}] {prompts[idx][:70]}...")
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Pipeline
    print("\n" + "-" * 70)
    print("TEST 2: Pipeline (with YOLO + Hierarchy)")
    print("-" * 70)
    
    try:
        from services.stable_food_pipeline import get_stable_pipeline
        
        pipeline = get_stable_pipeline()
        result = pipeline.recognize_pil(image, top_k_dishes=5)
        
        print(f"\nFood Group: {result.food_group.name}")
        print(f"Confidence: {result.food_group.confidence:.3f}")
        print(f"Cuisine: {result.food_group.cuisine}")
        
        if result.dish_suggestions:
            print("\nTop Dish Suggestions:")
            for i, dish in enumerate(result.dish_suggestions[:5]):
                print(f"  {i+1}. [{dish.confidence:.3f}] {dish.name}")
        
        best_dish = result.get_best_dish()
        print(f"\nBest Dish (after gate): {best_dish if best_dish else 'None (rejected)'}")
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Pipeline with YOLO Bypass
    print("\n" + "-" * 70)
    print("TEST 3: Pipeline (YOLO BYPASSED - Full Image)")
    print("-" * 70)
    
    try:
        from services.stable_food_pipeline import get_stable_pipeline
        
        pipeline = get_stable_pipeline()
        result = pipeline.recognize_pil(image, top_k_dishes=5, bypass_yolo=True)
        
        print(f"\nFood Group: {result.food_group.name}")
        print(f"Confidence: {result.food_group.confidence:.3f}")
        print(f"Cuisine: {result.food_group.cuisine}")
        
        if result.dish_suggestions:
            print("\nTop Dish Suggestions:")
            for i, dish in enumerate(result.dish_suggestions[:5]):
                print(f"  {i+1}. [{dish.confidence:.3f}] {dish.name}")
        
        best_dish = result.get_best_dish()
        print(f"\nBest Dish (after gate): {best_dish if best_dish else 'None (rejected)'}")
        
    except Exception as e:
        print(f"❌ Bypass test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided an image
        custom_image = sys.argv[1]
        print(f"Using custom image: {custom_image}")
        # TODO: Add custom image support
    else:
        test_with_sample_image()
