import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.continental_retrieval import get_continental_retrieval_system

def run_test():
    print("=== Continental Retrieval System Test ===")
    
    start_init = time.time()
    system = get_continental_retrieval_system()
    init_duration = time.time() - start_init
    
    print(f"Initialization (Load + Index) took: {init_duration:.2f} seconds")
    
    # Check if text index is built
    if system.text_features is not None:
        print(f"Text index shape: {system.text_features.shape}")
    else:
        print("Error: Text index not built!")
        return

    # test with a dummy image (random noise) to verify the flow
    from PIL import Image
    import torch
    
    dummy_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    
    print("\nRunning inference on dummy image...")
    start_inf = time.time()
    result = system.main_inference(dummy_img, k=5)
    inf_duration = time.time() - start_inf
    
    print(f"Inference took: {inf_duration:.4f} seconds")
    print("\nResults:")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    if result['top_k_predictions']:
        print("\nTop 5 Predictions:")
        for i, pred in enumerate(result['top_k_predictions'], 1):
            print(f"  {i}. {pred['dish']} (Score: {pred['score']:.4f})")

    if result['status'] == 'unknown':
        print("\nNote: Status is 'unknown' as expected for a noise image (threshold: 0.20)")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Test failed with error: {e}")
