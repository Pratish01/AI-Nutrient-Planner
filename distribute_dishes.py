
import os
import shutil

# Categories and their keywords
CATEGORIES = {
    "dishes_dal.txt": ["dal", "lentil", "sambhar", "kootu", "rasam", "amti", "kuzhambu", "tadka", "makhani"],
    "dishes_rice_dish.txt": ["rice", "biryani", "pulao", "khichdi", "fried rice", "bath", "tahri", "pongal", "sadham"],
    "dishes_indian_bread.txt": ["roti", "naan", "paratha", "kulcha", "chapati", "phulka", "bhakri", "thepla", "poori", "puri", "bhatura"],
    "dishes_street_food.txt": ["chaat", "pani puri", "bhel", "sev", "samosa", "kachori", "vada pav", "dabelli", "tikki", "pakora", "bhaji", "roll", "frankie", "momos"],
    "dishes_south_indian.txt": ["idli", "dosa", "vada", "uttapam", "appam", "paniyaram", "upma"],
    "dishes_dessert.txt": ["halwa", "kheer", "jamun", "rasgulla", "payasam", "mysore pak", "barfi", "ladoo", "laddu", "peda", "jalebi"],
    "dishes_other.txt": [] # Default for everything else
}

def distribute():
    base_dir = "data"
    master_file = os.path.join(base_dir, "expanded_indian_food_3000_plus.txt")
    
    if not os.path.exists(master_file):
        print(f"Master file not found: {master_file}")
        return

    print("Reading master file...")
    with open(master_file, 'r', encoding='utf-8') as f:
        all_dishes = [line.strip() for line in f if line.strip()]

    categorized = {k: [] for k in CATEGORIES.keys()}
    
    print(f"Distributing {len(all_dishes)} dishes...")
    
    for dish in all_dishes:
        dish_lower = dish.lower()
        assigned = False
        
        # Check specific categories first
        for filename, keywords in CATEGORIES.items():
            if filename == "dishes_other.txt": continue
            
            for kw in keywords:
                if kw in dish_lower:
                    categorized[filename].append(dish)
                    assigned = True
                    break
            if assigned: break
        
        # If not assigned to specific, add to other (Curries, Sabzis usually fall here)
        if not assigned:
            categorized["dishes_other.txt"].append(dish)

    # Write files
    for filename, dishes in categorized.items():
        if not dishes: continue
        
        filepath = os.path.join(base_dir, filename)
        
        # Remove duplicates and sort
        unique_dishes = sorted(list(set(dishes)))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(unique_dishes))
            
        print(f"Written {len(unique_dishes)} items to {filename}")

if __name__ == "__main__":
    distribute()
