"""
Download sample Indian food images for testing the food recognition system.
"""

import os
import requests
from pathlib import Path

def download_image(url, filename):
    """Download an image from URL to test_images folder."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    filepath = test_dir / filename
    
    if filepath.exists():
        print(f"✓ {filename} already exists")
        return True
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def download_test_images():
    """Download a set of Indian food images for testing."""
    
    # Sample Indian food images (using placeholder/sample URLs)
    # You can replace these with actual food images from your phone or camera
    
    images = {
        # Format: "filename.jpg": "url"
        # Note: These are placeholder URLs - replace with real food images
        
        # For now, create placeholders and instructions
    }
    
    print("=" * 70)
    print("🍛 INDIAN FOOD TEST IMAGE SETUP")
    print("=" * 70)
    
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print(f"\nTest images directory created: {test_dir.absolute()}")
    
    # Check if user already has images
    existing = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if existing:
        print(f"\n✓ Found {len(existing)} existing images:")
        for img in existing:
            print(f"  - {img.name}")
        print("\nYou can test with these using:")
        print(f"  python quick_accuracy_test.py test_images/{existing[0].name}")
    else:
        print("\n📸 To add test images:")
        print("1. Take photos of Indian food with your phone/camera")
        print("2. Save them to the 'test_images' folder")
        print("3. Or copy existing food images to 'test_images' folder")
        
        print("\n🌐 Or download from the web:")
        print("You can download food images from:")
        print("  - Google Images (search 'naan bread', 'pani puri', etc.)")
        print("  - Food websites")
        print("  - Your own photo library")
    
    print("\n" + "=" * 70)
    print("SAMPLE INDIAN FOOD TO TEST")
    print("=" * 70)
    
    test_categories = [
        ("Naan/Roti", "Test: Indian flatbread accuracy"),
        ("Pani Puri", "Test: Street food detection and relaxed gate"),
        ("Biryani", "Test: Rice dish classification"),
        ("Dal/Curry", "Test: Liquid curry recognition"),
        ("Dosa/Idli", "Test: South Indian breakfast"),
        ("Gulab Jamun", "Test: Indian dessert classification"),
    ]
    
    for i, (food, purpose) in enumerate(test_categories, 1):
        print(f"{i}. {food:20s} - {purpose}")
    
    print("\n" + "=" * 70)
    print("TESTING INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Add your food images to 'test_images' folder")
    print("2. Run: python quick_accuracy_test.py test_images/your_image.jpg")
    print("3. Check the food group confidence and best dish")
    print("\nExpected accuracy after all fixes:")
    print("  - Food Group: 0.60-0.90 confidence")
    print("  - Dish: Should resolve to specific dish name")

if __name__ == "__main__":
    download_test_images()
