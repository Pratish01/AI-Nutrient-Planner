from typing import Dict, List, Any, Optional
from services.nutrition_registry import get_nutrition_registry

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default nutritional targets (can be customized per condition)
DEFAULT_TARGETS = {
    "calories": 700,      # Per meal
    "protein_g": 25,      # Minimum per meal
    "carbs_g": 60,        # Maximum per meal
    "sugar_g": 15,        # Maximum per meal
    "sodium_mg": 600,     # Maximum per meal
    "fiber_g": 8,         # Minimum per meal
}

# Condition-specific targets
CONDITION_TARGETS = {
    "diabetes": {
        "sugar_g": 10,
        "carbs_g": 45,
    },
    "hypertension": {
        "sodium_mg": 400,
    },
    "obesity": {
        "calories": 500,
        "fat_g": 15,
    },
    "kidney": {
        "sodium_mg": 350,
        "protein_g": 20,  # Maximum for kidney disease
    }
}


class MealFixService:
    """
    Service for analyzing meals and providing fix suggestions.
    Uses centralized registry for food data.
    """
    
    def __init__(self):
        """Initialize the meal fix service."""
        self.registry = get_nutrition_registry()
        self.food_db = self.registry.get_all()  # Still need the list for some logic
    
    def analyze_meal(
        self,
        detected_items: List[Dict[str, Any]],
        user_conditions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a meal and provide fix suggestions.
        """
        if not self.food_db:
            return {
                "success": False,
                "error": "Nutrition database not available",
                "analysis": {},
                "suggestions": []
            }
        
        # Get targets based on conditions
        targets = self._get_targets_for_conditions(user_conditions or [])
        
        # Calculate meal totals
        analysis = self._analyze_nutrition(detected_items)
        
        # Generate fix suggestions
        suggestions = self._generate_suggestions(analysis, targets, detected_items, user_conditions)
        
        # Determine verdict
        if not suggestions:
            verdict = "Healthy"
            message = "✅ Great meal! Good nutritional balance."
        else:
            problem_count = len([s for s in suggestions if s.get("type") == "warning"])
            verdict = "Needs Fix"
            message = f"Found {problem_count} nutritional issue(s) that could be improved."
        
        return {
            "success": True,
            "verdict": verdict,
            "message": message,
            "analysis": analysis,
            "targets": targets,
            "suggestions": suggestions,
            "items_analyzed": len(detected_items)
        }
    
    def _get_targets_for_conditions(self, conditions: List[str]) -> Dict[str, float]:
        """Get nutritional targets based on user's health conditions."""
        targets = DEFAULT_TARGETS.copy()
        
        for condition in conditions:
            condition_lower = condition.lower()
            for key, overrides in CONDITION_TARGETS.items():
                if key in condition_lower:
                    for nutrient, value in overrides.items():
                        if nutrient in targets:
                            if nutrient in ["calories", "sugar_g", "sodium_mg", "carbs_g", "fat_g"]:
                                targets[nutrient] = min(targets[nutrient], value)
                            elif nutrient == "protein_g" and "kidney" in condition_lower:
                                targets[nutrient] = value
        
        return targets
    
    def _analyze_nutrition(self, detected_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze total nutrition of detected items."""
        totals = {
            "calories": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0,
            "sugar_g": 0, "fiber_g": 0, "sodium_mg": 0,
        }
        
        item_details = []
        for item in detected_items:
            name = item.get("name", "")
            grams = item.get("grams", 100)
            
            # Use provided nutrition or find in DB
            nutrition = item.get("nutrition_override")
            if not nutrition:
                nutrition = self._find_food_nutrition(name)
            
            if nutrition:
                scale = 1.0 if "_scale_factor" in nutrition else grams / 100
                
                item_nutrition = {}
                for key in totals:
                    val = float(nutrition.get(key, 0) or 0) * scale
                    totals[key] += val
                    item_nutrition[key] = round(val, 1)
                
                item_details.append({
                    "name": name, "grams": grams,
                    "nutrition": item_nutrition, "found_in_db": True
                })
            else:
                item_details.append({
                    "name": name, "grams": grams,
                    "nutrition": None, "found_in_db": False
                })
        
        for key in totals:
            totals[key] = round(totals[key], 1)
            
        return {"totals": totals, "items": item_details}
    
    def _find_food_nutrition(self, food_name: str) -> Optional[Dict[str, Any]]:
        """Find nutrition data using fuzzy name matching."""
        if not self.food_db: return None
        food_lower = food_name.lower()
        
        # 1. Exact match
        for row in self.food_db:
            if row.get('Dish Name', '').strip().lower() == food_lower:
                return self._extract_nutrition_from_row(row)
        
        # 2. Contains match
        for row in self.food_db:
            if food_lower in row.get('Dish Name', '').lower():
                return self._extract_nutrition_from_row(row)
        
        # 3. Word match
        words = [w for w in food_lower.split() if len(w) > 3]
        for word in words:
            for row in self.food_db:
                if word in row.get('Dish Name', '').lower():
                    return self._extract_nutrition_from_row(row)
                    
        return None
    
    def _extract_nutrition_from_row(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Extract and clean nutrition values from a CSV row."""
        try:
            return {
                "calories": float(row.get("Calories (kcal)", 0) or 0),
                "protein_g": float(row.get("Protein (g)", 0) or 0),
                "carbs_g": float(row.get("Carbohydrates (g)", 0) or 0),
                "fat_g": float(row.get("Fats (g)", 0) or 0),
                "sugar_g": float(row.get("Free Sugar (g)", 0) or 0),
                "fiber_g": float(row.get("Fibre (g)", 0) or 0),
                "sodium_mg": float(row.get("Sodium (mg)", 0) or 0),
            }
        except (ValueError, TypeError):
            return {}

    def _generate_suggestions(
        self,
        analysis: Dict[str, Any],
        targets: Dict[str, float],
        detected_items: List[Dict[str, Any]],
        conditions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate fix suggestions based on analysis."""
        suggestions = []
        totals = analysis.get("totals", {})
        conditions = [c.lower() for c in (conditions or [])]
        
        # Check PROTEIN
        if totals.get("protein_g", 0) < targets.get("protein_g", 25):
            deficit = targets["protein_g"] - totals.get("protein_g", 0)
            high_protein = self._find_high_protein_options()
            suggestions.append({
                "type": "warning", "category": "protein", "title": "Low Protein",
                "message": f"Your meal has only {totals.get('protein_g', 0):.0f}g protein. You need {deficit:.0f}g more.",
                "icon": "⚠️", "fix": f"Add a serving of **{high_protein}** to boost protein."
            })
        
        # Check CARBS
        if totals.get("carbs_g", 0) > targets.get("carbs_g", 60):
            excess = totals.get("carbs_g", 0) - targets["carbs_g"]
            culprit = "starchy foods"
            for item in detected_items:
                name = item.get("name", "").lower()
                if any(k in name for k in ["rice", "roti", "naan", "paratha", "bread", "pasta"]):
                    culprit = item.get("name")
                    break
            suggestions.append({
                "type": "warning", "category": "carbs", "title": "High Carbohydrates",
                "message": f"Your meal has {totals.get('carbs_g', 0):.0f}g carbs ({excess:.0f}g over limit).",
                "icon": "⚠️", "fix": f"Reduce **{culprit}** portion by half to cut ~{excess/2:.0f}g carbs."
            })

        # SUGAR, SODIUM, CALORIES, FIBER checks
        self._add_threshold_suggestion(suggestions, totals, targets, "sugar_g", "Sugar", "🍬", conditions, "diabetes")
        self._add_threshold_suggestion(suggestions, totals, targets, "sodium_mg", "Sodium", "🧂", conditions, "hypertension")
        self._add_threshold_suggestion(suggestions, totals, targets, "calories", "Calories", "📊", conditions, "obesity")
        
        if totals.get("fiber_g", 0) < targets.get("fiber_g", 5):
            suggestions.append({
                "type": "info", "category": "fiber", "title": "Low Fiber", "icon": "🥗",
                "message": f"Only {totals.get('fiber_g', 0):.0f}g fiber. Adding more helps digestion.",
                "fix": "Add a side salad or include leafy greens."
            })
        
        return suggestions

    def _add_threshold_suggestion(self, suggestions, totals, targets, key, label, icon, conditions, condition_keyword):
        limit = targets.get(key)
        val = totals.get(key, 0)
        if limit and val > limit:
            is_critical = any(condition_keyword in c for c in conditions)
            suggestions.append({
                "type": "warning" if is_critical else "info",
                "category": key.split('_')[0], "title": f"High {label}",
                "message": f"{label} is {val:.0f} (budget: {limit:.0f}).",
                "icon": icon, "fix": f"Choose lower {label.lower()} options or reduce portion."
            })

    def _find_high_protein_options(self) -> str:
        """Find a high-protein food option using list filtering."""
        if not self.food_db: return "boiled eggs or paneer"
        
        try:
            # Filter for high protein
            candidates = [r for r in self.food_db if float(r.get('Protein (g)', 0) or 0) > 10 and float(r.get('Calories (kcal)', 0) or 0) < 300]
            if not candidates: return "eggs or paneer"
            
            # Sort by protein-to-calorie ratio
            candidates.sort(key=lambda r: float(r.get('Protein (g)', 1))/max(1, float(r.get('Calories (kcal)', 1))), reverse=True)
            
            # Prioritize common ones
            keywords = ['egg', 'paneer', 'dal', 'chicken', 'fish', 'curd']
            for kw in keywords:
                for r in candidates:
                    if kw in r.get('Dish Name', '').lower():
                        return r['Dish Name']
            return candidates[0]['Dish Name']
        except:
            return "boiled eggs or paneer"


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_meal_fix_service: Optional[MealFixService] = None

def get_meal_fix_service() -> MealFixService:
    global _meal_fix_service
    if _meal_fix_service is None:
        _meal_fix_service = MealFixService()
    return _meal_fix_service

def fix_meal(detected_items: List[Dict[str, Any]], user_conditions: List[str] = None) -> Dict[str, Any]:
    return get_meal_fix_service().analyze_meal(detected_items, user_conditions)
