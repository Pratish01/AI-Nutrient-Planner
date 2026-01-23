"""
Diagnostic script to compare OpenCLIP results in standalone vs pipeline mode.
This helps identify why the pipeline gives different results than standalone CLIP.
"""

import sys
import os
from pathlib import Path
from PIL import Image
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_standalone_clip(image_path: str):
    """Test OpenCLIP in standalone mode (direct classification)."""
    print("\n" + "="*70)
    print("🔍 STANDALONE OpenCLIP TEST")
    print("="*70)
    
    try:
        import open_clip
        
        # Load model
        print(f"Loading ViT-B-16 with laion2b_s34b_b88k...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained='laion2b_s34b_b88k'
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model loaded on {device}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
        
        # Load food group prompts
        prompts_path = Path("data/CLIP_Food_Groups.txt")
        prompts = []
        with open(prompts_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
        
        print(f"✓ Loaded {len(prompts)} food group prompts")
        
        # Encode image
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Encode text prompts
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        similarities = (image_embedding @ text_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0)
        
        # Sort and display
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        
        print("\n📊 RESULTS (Top 10):")
        print("-" * 70)
        for i, (prob, idx) in enumerate(zip(sorted_probs[:10], sorted_idx[:10])):
            print(f"  {i+1}. {prob.item():.3f} - {prompts[idx.item()][:60]}...")
        
        return sorted_probs[0].item(), prompts[sorted_idx[0].item()]
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_pipeline_clip(image_path: str):
    """Test OpenCLIP through the pipeline (with YOLO, hierarchy, etc)."""
    print("\n" + "="*70)
    print("🏭 PIPELINE OpenCLIP TEST")
    print("="*70)
    
    try:
        from services.stable_food_pipeline import get_stable_pipeline
        
        pipeline = get_stable_pipeline()
        print("✓ Pipeline loaded")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
        
        # Run recognition
        result = pipeline.recognize_pil(image, top_k_dishes=5)
        
        print("\n📊 RESULTS:")
        print("-" * 70)
        print(f"Food Group: {result.food_group.name}")
        print(f"Confidence: {result.food_group.confidence:.3f}")
        print(f"Cuisine: {result.food_group.cuisine}")
        
        if result.dish_suggestions:
            print("\nTop Dishes:")
            for i, dish in enumerate(result.dish_suggestions[:5]):
                print(f"  {i+1}. {dish.name}: {dish.confidence:.3f}")
        
        best_dish = result.get_best_dish()
        if best_dish:
            print(f"\n✓ Best Dish (after Specificity Gate): {best_dish}")
        else:
            print(f"\n⚠️ No dish passed Specificity Gate")
        
        return result.food_group.confidence, result.food_group.name
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_results(image_path: str):
    """Run both tests and compare results."""
    print("\n" + "="*70)
    print(f"🎯 COMPARING RESULTS FOR: {image_path}")
    print("="*70)
    
    standalone_conf, standalone_pred = test_standalone_clip(image_path)
    pipeline_conf, pipeline_pred = test_pipeline_clip(image_path)
    
    print("\n" + "="*70)
    print("📈 COMPARISON")
    print("="*70)
    print(f"Standalone: {standalone_pred} ({standalone_conf:.3f})")
    print(f"Pipeline:   {pipeline_pred} ({pipeline_conf:.3f})")
    
    if standalone_pred and pipeline_pred:
        if standalone_pred != pipeline_pred:
            print("\n⚠️  MISMATCH DETECTED!")
            print("   → Check YOLO cropping, preprocessing, or prompt differences")
        else:
            print("\n✓ Results match!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        image_path = "test_images/pani_puri.jpg"
        print(f"No image specified. Using default: {image_path}")
        print("Usage: python diagnose_clip_accuracy.py <path_to_image>\n")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("\nPlease provide a valid image path.")
        sys.exit(1)
    
    compare_results(image_path)
