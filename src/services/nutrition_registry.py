"""
Centralized Nutrition Database Registry
Unifies nutrition data loading to save memory and ensure consistency.
Supports Cuisine and Food_Group for hierarchical classification.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class NutritionRegistry:
    """
    Singleton registry for nutrition data.
    Loads the CSV once and provides access to all services.
    Supports hierarchical lookup by Cuisine and Food_Group.
    """
    _instance = None
    _data: List[Dict[str, Any]] = []
    _indexed_data: Dict[str, Dict[str, Any]] = {}
    # Index by (cuisine, food_group) for fast hierarchical lookup
    _cuisine_food_group_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    # Cache for average nutrition by cuisine+food_group
    _avg_nutrition_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NutritionRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only load once
        if not self._data:
            self._load_database()
            
    def _normalize_key(self, s: str) -> str:
        """Normalize string key: lower, strip, remove spaces around slashes."""
        if not s: return ""
        # "Dessert / Sweet" -> "dessert/sweet"
        return s.lower().strip().replace(" / ", "/").replace("/ ", "/").replace(" /", "/")

    def _load_database(self):
        """Load the nutrition database from available CSV sources."""
        data_dir = Path(__file__).parent.parent.parent / "data"
        
        # Priority list of CSV sources - prefer files with Cuisine/Food_Group columns
        src_candidates = [
            data_dir / "FINAL_ACCURATE_FOOD_DATASET_WITH_CUISINE (1).csv",
            data_dir / "FINAL_ACCURATE_FOOD_DATASET_WITH_CUISINE.csv",
            data_dir / "Indian_Continental_Nutrition_With_Dal_Variants.csv",
            data_dir / "Indian_Continental_Nutrition_With_Density.csv",
            data_dir / "FINAL_MERGED_INDIAN_USDA_WITH_DENSITY.csv",
            data_dir / "Indian_Food_Nutrition_Processed.csv",
            data_dir / "healthy_eating_dataset.csv",
            data_dir / "sample_foods.csv"
        ]
        
        csv_path = next((p for p in src_candidates if p.exists()), None)
            
        try:
            if csv_path:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Normalize keys for consistency across different CSV formats
                        name = (row.get('Dish Name') or row.get('meal_name') or row.get('name', '')).strip()
                        if not name: continue
                        
                        # Get Cuisine and Food_Group (for hierarchical classification)
                        cuisine = (row.get('Cuisine') or '').strip()
                        food_group = (row.get('Food_Group') or '').strip()
                        
                        # Store standardized entry with Cuisine/Food_Group
                        item = {
                            'name': name,
                            'cuisine': cuisine,
                            'food_group': food_group,
                            'calories': float(row.get('Calories (kcal)', 0) or row.get('calories', 0) or 0),
                            'protein_g': float(row.get('Protein (g)', 0) or row.get('protein_g', 0) or 0),
                            'carbs_g': float(row.get('Carbohydrates (g)', 0) or row.get('carbs_g', 0) or 0),
                            'fat_g': float(row.get('Fats (g)', 0) or row.get('fat_g', 0) or 0),
                            'sugar_g': float(row.get('Free Sugar (g)', 0) or row.get('sugar_g', 0) or 0),
                            'sodium_mg': float(row.get('Sodium (mg)', 0) or row.get('sodium_mg', 0) or 0),
                            'fiber_g': float(row.get('Fibre (g)', 0) or row.get('fiber_g', 0) or 0),
                            'density': float(row.get('Density (g/cm3)', 0) or row.get('density', 1.0) or 1.0),
                        }
                        
                        self._data.append(item)
                        self._indexed_data[name.lower()] = item
                        
                        # Build cuisine+food_group index
                        if cuisine and food_group:
                            # Normalize key using helper
                            key = (self._normalize_key(cuisine), self._normalize_key(food_group))
                            if key not in self._cuisine_food_group_index:
                                self._cuisine_food_group_index[key] = []
                            self._cuisine_food_group_index[key].append(item)
                
                print(f"[Registry] Loaded {len(self._data)} items from {csv_path.name}")
                print(f"[Registry] Built index for {len(self._cuisine_food_group_index)} cuisine+food_group combinations")
            else:
                print(f"[Registry] WARNING: No nutrition database found!")
        except Exception as e:
            print(f"[Registry] Error loading database: {e}")
            
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all raw food items."""
        return self._data
        
    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get nutrition for an item by name with fuzzy fallback."""
        if not name: return None
        
        # 1. Exact match
        clean_name = name.lower().strip()
        result = self._indexed_data.get(clean_name)
        if result: return result
        
        # 2. Fuzzy fallback
        fuzzy_results = self.fuzzy_search(name, limit=1)
        if fuzzy_results:
            print(f"[Registry] Exact match failed for '{name}', using fuzzy: '{fuzzy_results[0]['name']}'")
            return fuzzy_results[0]
            
        return None
        
    def fuzzy_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find items matching the query."""
        query = query.lower().strip()
        results = []
        
        # 1. Start with exact/starts-with
        for name, row in self._indexed_data.items():
            if name == query or name.startswith(query):
                results.append(row)
                if len(results) >= limit: return results
                
        # 2. Contains
        for name, row in self._indexed_data.items():
            if query in name and row not in results:
                results.append(row)
                if len(results) >= limit: return results
                
        return results
    
    def get_by_cuisine_and_food_group(self, cuisine: str, food_group: str) -> Optional[Dict[str, float]]:
        """
        Get average nutrition for items matching cuisine and food_group.
        Used for hierarchical classification nutrition lookup.
        
        Args:
            cuisine: Cuisine type (e.g., 'Indian', 'Continental')
            food_group: Food group (e.g., 'Dal', 'Rice Dish', 'Pizza')
            
        Returns:
            Dict with average nutrition values for per 100g, or None if not found
        """
        key = (self._normalize_key(cuisine), self._normalize_key(food_group))
        
        # Check cache first
        if key in self._avg_nutrition_cache:
            return self._avg_nutrition_cache[key]
        
        # Get items matching this combination
        items = self._cuisine_food_group_index.get(key, [])
        
        if not items:
            print(f"[Registry] No items found for {cuisine}/{food_group}")
            return None
        
        # Compute averages
        avg = {
            'calories': sum(i['calories'] for i in items) / len(items),
            'protein_g': sum(i['protein_g'] for i in items) / len(items),
            'carbs_g': sum(i['carbs_g'] for i in items) / len(items),
            'fat_g': sum(i['fat_g'] for i in items) / len(items),
            'sugar_g': sum(i['sugar_g'] for i in items) / len(items),
            'sodium_mg': sum(i['sodium_mg'] for i in items) / len(items),
            'fiber_g': sum(i['fiber_g'] for i in items) / len(items),
            'density': sum(i['density'] for i in items) / len(items),
            'item_count': len(items),
            'cuisine': cuisine,
            'food_group': food_group
        }
        
        # Cache result
        self._avg_nutrition_cache[key] = avg
        return avg
    
    def get_unique_food_groups_by_cuisine(self, cuisine: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get all unique food_groups, optionally filtered by cuisine.
        Used for generating CLIP prompts.
        
        Args:
            cuisine: Optional cuisine filter (e.g., 'Indian'). If None, returns all.
            
        Returns:
            Dict mapping cuisine -> list of food_groups
        """
        result: Dict[str, List[str]] = {}
        
        for (c, fg) in self._cuisine_food_group_index.keys():
            if cuisine and c != cuisine.lower():
                continue
            c_title = c.title()  # Normalize to title case
            if c_title not in result:
                result[c_title] = []
            fg_title = fg.title()
            if fg_title not in result[c_title]:
                result[c_title].append(fg_title)
        
        return result
    
    def get_all_cuisines(self) -> List[str]:
        """Get list of all unique cuisines in the database."""
        cuisines = set()
        for (c, _) in self._cuisine_food_group_index.keys():
            cuisines.add(c.title())
        return sorted(list(cuisines))
    
    def get_sample_items(self, cuisine: str, food_group: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample items for a cuisine/food_group combination."""
        key = (cuisine.lower().strip(), food_group.lower().strip())
        items = self._cuisine_food_group_index.get(key, [])
        return items[:limit]


# Singleton helper
_registry = None

def get_nutrition_registry() -> NutritionRegistry:
    """Access the global nutrition registry."""
    global _registry
    if _registry is None:
        _registry = NutritionRegistry()
    return _registry
