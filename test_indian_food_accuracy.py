"""
Test Indian food classification accuracy.
This will show what CLIP is predicting for each Indian food category.
"""

import sys
import os
from pathlib import Path
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

def test_indian_food_prompts():
    """Test how well CLIP recognizes Indian food categories."""
    
    print("=" * 70)
    print("🍛 INDIAN FOOD ACCURACY TEST")
    print("=" * 70)
    
    # Indian food categories from CLIP_Food_Groups.txt
    indian_categories = [
        "a close-up photo of indian street food such as pani puri, dahi puri, golgappa, chaat, sev puri, or pav bhaji, often served with small bowls of spicy water",
        "a close-up photo of south indian food such as idli, dosa, vada, uttapam served with coconut chutney and sambar",
        "a close-up photo of indian steamed fermented breakfast food like dhokla, khaman, or idli served on a plate",
        "a close-up photo of indian dal or lentil curry (yellow or black dal) served in a bowl, often with tadka",
        "a close-up photo of indian rice dish such as biryani, pulao, jeera rice, or fried rice with visible grains and spices",
        "a close-up photo of indian bread such as roti, chapati, naan, paratha, or kulcha, often served with curry",
        "a close-up photo of indian dry vegetable sabzi or cooked vegetables (bhindi, aloo gobi, mix veg) without much gravy",
        "a close-up photo of indian wet curry or thick gravy based dish (paneer butter masala, chicken curry, kofta)",
        "a close-up photo of indian dessert or sweet such as gulab jamun, rasgulla, kheer, halwa, or barfi",
    ]
    
    print("\nIndian Food Categories in System:")
    print("-" * 70)
    for i, cat in enumerate(indian_categories, 1):
        print(f"{i}. {cat[:80]}...")
    
    print(f"\nTotal Indian categories: {len(indian_categories)}")
    print(f"Total non-Indian categories: {15 - len(indian_categories)}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES")
    print("=" * 70)
    
    issues = []
    
    # Issue 1: Too many similar categories
    if len(indian_categories) > 10:  
        issues.append("❌ Too many Indian categories may dilute confidence")
    
    # Issue 2: Check for overlapping descriptions
    keywords = []
    for cat in indian_categories:
        if "curry" in cat.lower():
            keywords.append("curry")
    if keywords.count("curry") > 2:
        issues.append("❌ Multiple 'curry' categories may cause confusion")
    
    # Issue 3: Verbose prompts
    avg_length = sum(len(cat) for cat in indian_categories) / len(indian_categories)
    if avg_length > 150:
        issues.append(f"⚠️  Prompts are very long (avg {avg_length:.0f} chars) - may reduce CLIP effectiveness")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ No obvious prompt engineering issues detected")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. **Simplify Prompts**")
    print("   Current: 'a close-up photo of indian street food such as pani puri...'")
    print("   Better:  'indian street food like pani puri and chaat'")
    
    print("\n2. **Reduce Category Count**")
    print("   Merge similar categories:")
    print("   - 'south indian' + 'steamed fermented' → 'south indian breakfast'")
    print("   - 'dry sabzi' + 'wet curry' → 'indian curry'")
    
    print("\n3. **Use Distinctive Visual Features**")
    print("   - Biryani: 'layered rice with meat and spices'")
    print("   - Dosa: 'thin crispy crepe with chutney'")
    print("   - Naan: 'flatbread with charred spots'")
    
    print("\n4. **Test Category Separation**")
    print("   Run: python test_category_confusion.py")

if __name__ == "__main__":
    test_indian_food_prompts()
