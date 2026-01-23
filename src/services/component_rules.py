"""
Component-Driven Food Recognition Engine

This module implements deterministic, rule-based Indian food recognition.
Dish names are DERIVED from visual components, never predicted directly.

Pipeline: Components → Structure → Rules → Dish (if justified)
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# VISUAL COMPONENTS (Binary Detection)
# =============================================================================

class VisualComponent(Enum):
    """Visual components that can be detected in food images."""
    # Structural components
    HOLLOW_SHELL = "hollow_shell"                 # Crispy shell with visible empty cavity
    SOLID_FRIED_PATTY = "solid_fried_patty"       # Dense fried ball/patty, no hollow
    FLAT_BREAD = "flat_bread"                     # Roti, naan, paratha
    LONG_GRAIN_RICE = "long_grain_rice"           # Biryani, pulao style
    SHORT_GRAIN_RICE = "short_grain_rice"         # Idli batter, kheer style
    
    # Protein components
    PANEER_CUBES = "paneer_cubes"
    CHICKEN_PIECES = "chicken_pieces"
    MUTTON_PIECES = "mutton_pieces"
    EGG_VISIBLE = "egg_visible"
    
    # Legume components
    YELLOW_LENTILS = "yellow_lentils"             # Moong, arhar
    DARK_LENTILS = "dark_lentils"                 # Urad, masoor
    CHICKPEAS = "chickpeas"
    
    # Gravy/Liquid components
    THICK_CREAMY_GRAVY = "thick_creamy_gravy"     # Butter masala style
    WATERY_GRAVY = "watery_gravy"                 # Dal tadka style
    SPICED_LIQUID = "spiced_liquid"               # Pani puri water
    COCONUT_CHUTNEY = "coconut_chutney"
    SAMBAR = "sambar"
    
    # Texture indicators
    PUFFED_RICE = "puffed_rice"                   # Murmura, bhel
    SEV_NOODLES = "sev_noodles"                   # Thin fried noodles
    FRIED_TEXTURE = "fried_texture"
    GRILLED_CHARRED = "grilled_charred"
    STEAMED_TEXTURE = "steamed_texture"
    
    # Scale indicators
    SMALL_SNACK_SCALE = "small_snack_scale"       # Street food portion
    PLATED_MEAL_SCALE = "plated_meal_scale"       # Full meal
    HAND_HELD_SCALE = "hand_held_scale"           # Samosa, vada pav
    
    # Garnish/Toppings
    CURD_TOPPING = "curd_topping"                 # Dahi vada, raita
    TAMARIND_SAUCE = "tamarind_sauce"
    GREEN_CHUTNEY = "green_chutney"
    ONION_TOPPING = "onion_topping"


# CLIP prompts for each component (used for binary detection)
COMPONENT_PROMPTS: Dict[VisualComponent, str] = {
    VisualComponent.HOLLOW_SHELL: "a crispy hollow puri shell with visible empty cavity",
    VisualComponent.SOLID_FRIED_PATTY: "a solid dense fried ball or patty with no hollow center",
    VisualComponent.FLAT_BREAD: "indian flatbread like roti naan or paratha",
    VisualComponent.LONG_GRAIN_RICE: "long grain basmati rice with separate grains",
    VisualComponent.SHORT_GRAIN_RICE: "short grain sticky rice",
    VisualComponent.PANEER_CUBES: "white paneer cheese cubes in curry",
    VisualComponent.CHICKEN_PIECES: "cooked chicken pieces in curry or gravy",
    VisualComponent.MUTTON_PIECES: "cooked mutton or lamb pieces",
    VisualComponent.EGG_VISIBLE: "visible boiled or fried egg",
    VisualComponent.YELLOW_LENTILS: "yellow colored lentils or dal",
    VisualComponent.DARK_LENTILS: "dark brown or black lentils like urad dal",
    VisualComponent.CHICKPEAS: "chickpeas or chana",
    VisualComponent.THICK_CREAMY_GRAVY: "thick creamy orange or red gravy",
    VisualComponent.WATERY_GRAVY: "thin watery dal or soup",
    VisualComponent.SPICED_LIQUID: "spiced flavored water in small bowl",
    VisualComponent.COCONUT_CHUTNEY: "white coconut chutney",
    VisualComponent.SAMBAR: "orange sambar with vegetables",
    VisualComponent.PUFFED_RICE: "puffed rice or murmura",
    VisualComponent.SEV_NOODLES: "thin crispy sev noodles",
    VisualComponent.FRIED_TEXTURE: "deep fried crispy food texture",
    VisualComponent.GRILLED_CHARRED: "grilled or charred marks on food",
    VisualComponent.STEAMED_TEXTURE: "steamed soft food texture",
    VisualComponent.SMALL_SNACK_SCALE: "small snack sized food portions",
    VisualComponent.PLATED_MEAL_SCALE: "full meal on a plate",
    VisualComponent.HAND_HELD_SCALE: "hand held snack food",
    VisualComponent.CURD_TOPPING: "white curd or yogurt topping",
    VisualComponent.TAMARIND_SAUCE: "brown tamarind sauce or chutney",
    VisualComponent.GREEN_CHUTNEY: "green mint or coriander chutney",
    VisualComponent.ONION_TOPPING: "chopped onion topping",
}


# =============================================================================
# STRUCTURE TYPES
# =============================================================================

class FoodStructure(Enum):
    """Composition structure of the food."""
    DRY_ONLY = "dry_only"
    LIQUID_ONLY = "liquid_only"
    DRY_PLUS_LIQUID = "dry_plus_liquid"
    MIXED_SCENE = "mixed_scene"  # Multiple small items
    SINGLE_ITEM = "single_item"
    UNKNOWN = "unknown"


# =============================================================================
# DISH DERIVATION RULES
# =============================================================================

@dataclass
class DishRule:
    """A deterministic rule for deriving a dish from components."""
    dish_name: str
    required_present: Set[VisualComponent]
    required_absent: Set[VisualComponent] = field(default_factory=set)
    priority: int = 0  # Higher = more specific, evaluated first
    food_group: str = "Other"
    cuisine: str = "Indian"


# Rules ordered by specificity (highest priority first)
DISH_RULES: List[DishRule] = [
    # =========================================================================
    # HOLLOW + LIQUID RULES (Highest Priority)
    # =========================================================================
    DishRule(
        dish_name="Pani Puri",
        required_present={
            VisualComponent.HOLLOW_SHELL,
            VisualComponent.SPICED_LIQUID,
            VisualComponent.SMALL_SNACK_SCALE,
        },
        required_absent={
            VisualComponent.SOLID_FRIED_PATTY,
        },
        priority=100,
        food_group="Street Food",
    ),
    DishRule(
        dish_name="Dahi Puri",
        required_present={
            VisualComponent.HOLLOW_SHELL,
            VisualComponent.CURD_TOPPING,
            VisualComponent.SMALL_SNACK_SCALE,
        },
        required_absent={
            VisualComponent.SOLID_FRIED_PATTY,
        },
        priority=95,
        food_group="Street Food",
    ),
    DishRule(
        dish_name="Sev Puri",
        required_present={
            VisualComponent.HOLLOW_SHELL,
            VisualComponent.SEV_NOODLES,
            VisualComponent.SMALL_SNACK_SCALE,
        },
        required_absent={
            VisualComponent.SOLID_FRIED_PATTY,
            VisualComponent.SPICED_LIQUID,
        },
        priority=90,
        food_group="Street Food",
    ),
    
    # =========================================================================
    # MIXED CHAAT RULES
    # =========================================================================
    DishRule(
        dish_name="Bhel Puri",
        required_present={
            VisualComponent.PUFFED_RICE,
            VisualComponent.SEV_NOODLES,
            VisualComponent.SMALL_SNACK_SCALE,
        },
        required_absent={
            VisualComponent.HOLLOW_SHELL,
        },
        priority=85,
        food_group="Street Food",
    ),
    DishRule(
        dish_name="Dahi Vada",
        required_present={
            VisualComponent.SOLID_FRIED_PATTY,
            VisualComponent.CURD_TOPPING,
        },
        required_absent={
            VisualComponent.HOLLOW_SHELL,
        },
        priority=80,
        food_group="Street Food",
    ),
    
    # =========================================================================
    # SOLID FRIED SNACK RULES
    # =========================================================================
    DishRule(
        dish_name="Sabudana Vada",
        required_present={
            VisualComponent.SOLID_FRIED_PATTY,
            VisualComponent.FRIED_TEXTURE,
        },
        required_absent={
            VisualComponent.HOLLOW_SHELL,
            VisualComponent.SPICED_LIQUID,
            VisualComponent.CURD_TOPPING,
        },
        priority=70,
        food_group="Street Food",
    ),
    DishRule(
        dish_name="Vada Pav",
        required_present={
            VisualComponent.SOLID_FRIED_PATTY,
            VisualComponent.FLAT_BREAD,
            VisualComponent.HAND_HELD_SCALE,
        },
        required_absent={
            VisualComponent.HOLLOW_SHELL,
        },
        priority=75,
        food_group="Street Food",
    ),
    DishRule(
        dish_name="Samosa",
        required_present={
            VisualComponent.FRIED_TEXTURE,
            VisualComponent.HAND_HELD_SCALE,
        },
        required_absent={
            VisualComponent.HOLLOW_SHELL,
            VisualComponent.SOLID_FRIED_PATTY,
        },
        priority=65,
        food_group="Street Food",
    ),
    
    # =========================================================================
    # CURRY RULES
    # =========================================================================
    DishRule(
        dish_name="Dal Makhani",
        required_present={
            VisualComponent.DARK_LENTILS,
            VisualComponent.THICK_CREAMY_GRAVY,
        },
        priority=60,
        food_group="Dal",
    ),
    DishRule(
        dish_name="Dal Tadka",
        required_present={
            VisualComponent.YELLOW_LENTILS,
            VisualComponent.WATERY_GRAVY,
        },
        priority=55,
        food_group="Dal",
    ),
    DishRule(
        dish_name="Paneer Butter Masala",
        required_present={
            VisualComponent.PANEER_CUBES,
            VisualComponent.THICK_CREAMY_GRAVY,
        },
        priority=60,
        food_group="Wet Curry",
    ),
    DishRule(
        dish_name="Butter Chicken",
        required_present={
            VisualComponent.CHICKEN_PIECES,
            VisualComponent.THICK_CREAMY_GRAVY,
        },
        priority=60,
        food_group="Wet Curry",
    ),
    
    # =========================================================================
    # SOUTH INDIAN RULES
    # =========================================================================
    DishRule(
        dish_name="Idli",
        required_present={
            VisualComponent.STEAMED_TEXTURE,
            VisualComponent.COCONUT_CHUTNEY,
        },
        required_absent={
            VisualComponent.FRIED_TEXTURE,
        },
        priority=50,
        food_group="South Indian",
    ),
    DishRule(
        dish_name="Dosa",
        required_present={
            VisualComponent.FRIED_TEXTURE,
            VisualComponent.COCONUT_CHUTNEY,
            VisualComponent.SAMBAR,
        },
        priority=55,
        food_group="South Indian",
    ),
    
    # =========================================================================
    # BREAD RULES
    # =========================================================================
    DishRule(
        dish_name="Naan",
        required_present={
            VisualComponent.FLAT_BREAD,
            VisualComponent.GRILLED_CHARRED,
        },
        priority=45,
        food_group="Indian Bread",
    ),
    DishRule(
        dish_name="Roti",
        required_present={
            VisualComponent.FLAT_BREAD,
        },
        required_absent={
            VisualComponent.GRILLED_CHARRED,
            VisualComponent.FRIED_TEXTURE,
        },
        priority=40,
        food_group="Indian Bread",
    ),
    
    # =========================================================================
    # RICE RULES
    # =========================================================================
    DishRule(
        dish_name="Biryani",
        required_present={
            VisualComponent.LONG_GRAIN_RICE,
            VisualComponent.PLATED_MEAL_SCALE,
        },
        priority=50,
        food_group="Rice Dish",
    ),
]

# Sort rules by priority (descending)
DISH_RULES.sort(key=lambda r: r.priority, reverse=True)


# =============================================================================
# COMPONENT RULE ENGINE
# =============================================================================

@dataclass
class ComponentDetectionResult:
    """Result of component detection step."""
    detected: Set[VisualComponent]
    confidences: Dict[VisualComponent, float]
    structure: FoodStructure


@dataclass
class DishDerivationResult:
    """Result of rule-based dish derivation."""
    detected_components: List[str]
    resolved_food: str
    resolution_type: str  # "component_rule" | "group" | "unknown"
    confidence: float
    matched_rule: Optional[str] = None
    food_group: str = "Unknown"
    cuisine: str = "Unknown"


class ComponentRuleEngine:
    """
    Deterministic rule engine for Indian food recognition.
    
    Pipeline:
    1. Detect visual components (binary)
    2. Analyze structure
    3. Evaluate rules in priority order
    4. Verify against database
    5. Return derived dish or fallback to group
    """
    
    def __init__(self, nutrition_registry=None):
        self.rules = DISH_RULES
        self.nutrition_registry = nutrition_registry
        
    def derive_dish(
        self,
        detected_components: Set[VisualComponent],
        structure: FoodStructure,
    ) -> DishDerivationResult:
        """
        Derive dish from detected components using deterministic rules.
        
        Args:
            detected_components: Set of detected visual components
            structure: Detected food structure
            
        Returns:
            DishDerivationResult with resolved food
        """
        # Validate mutual exclusivity
        if not self._validate_exclusivity(detected_components):
            logger.warning("Component exclusivity violation detected")
            return self._fallback_result(detected_components, "exclusivity_violation")
        
        # Evaluate rules in priority order
        for rule in self.rules:
            if self._rule_matches(rule, detected_components):
                # Verify dish exists in database
                if self._verify_in_database(rule.dish_name):
                    return DishDerivationResult(
                        detected_components=[c.value for c in detected_components],
                        resolved_food=rule.dish_name,
                        resolution_type="component_rule",
                        confidence=self._calculate_confidence(rule, detected_components),
                        matched_rule=rule.dish_name,
                        food_group=rule.food_group,
                        cuisine=rule.cuisine,
                    )
                else:
                    logger.info(f"Rule matched '{rule.dish_name}' but not in database")
        
        # No rule matched - fallback to group
        return self._fallback_result(detected_components, "no_rule_match")
    
    def _validate_exclusivity(self, components: Set[VisualComponent]) -> bool:
        """Validate mutually exclusive components."""
        hollow = VisualComponent.HOLLOW_SHELL in components
        solid = VisualComponent.SOLID_FRIED_PATTY in components
        
        if hollow and solid:
            logger.error("INVALID: Both hollow_shell and solid_fried_patty detected")
            return False
        return True
    
    def _rule_matches(self, rule: DishRule, components: Set[VisualComponent]) -> bool:
        """Check if a rule matches the detected components."""
        # All required present must be in components
        if not rule.required_present.issubset(components):
            return False
        
        # None of required absent should be in components
        if rule.required_absent.intersection(components):
            return False
        
        return True
    
    def _verify_in_database(self, dish_name: str) -> bool:
        """Verify dish exists in nutrition database."""
        if self.nutrition_registry is None:
            return True  # Skip check if no registry
        
        item = self.nutrition_registry.get_by_name(dish_name)
        return item is not None
    
    def _calculate_confidence(
        self,
        rule: DishRule,
        components: Set[VisualComponent]
    ) -> float:
        """Calculate confidence based on rule completeness."""
        required = len(rule.required_present)
        matched = len(rule.required_present.intersection(components))
        return matched / required if required > 0 else 0.0
    
    def _fallback_result(
        self,
        components: Set[VisualComponent],
        reason: str
    ) -> DishDerivationResult:
        """Generate fallback result when no rule matches."""
        # Try to infer food group from components
        food_group = self._infer_group(components)
        
        return DishDerivationResult(
            detected_components=[c.value for c in components],
            resolved_food=food_group,
            resolution_type="group",
            confidence=0.5,
            matched_rule=None,
            food_group=food_group,
            cuisine="Indian" if food_group != "Unknown" else "Unknown",
        )
    
    def _infer_group(self, components: Set[VisualComponent]) -> str:
        """Infer food group from components when no dish rule matches."""
        if VisualComponent.HOLLOW_SHELL in components or \
           VisualComponent.PUFFED_RICE in components or \
           VisualComponent.SMALL_SNACK_SCALE in components:
            return "Street Food"
        
        if VisualComponent.YELLOW_LENTILS in components or \
           VisualComponent.DARK_LENTILS in components:
            return "Dal"
        
        if VisualComponent.PANEER_CUBES in components or \
           VisualComponent.THICK_CREAMY_GRAVY in components:
            return "Wet Curry"
        
        if VisualComponent.FLAT_BREAD in components:
            return "Indian Bread"
        
        if VisualComponent.LONG_GRAIN_RICE in components:
            return "Rice Dish"
        
        if VisualComponent.COCONUT_CHUTNEY in components or \
           VisualComponent.SAMBAR in components:
            return "South Indian"
        
        return "Unknown"


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_rule_engine_instance: Optional[ComponentRuleEngine] = None

def get_component_rule_engine(nutrition_registry=None) -> ComponentRuleEngine:
    """Get or create the component rule engine singleton."""
    global _rule_engine_instance
    if _rule_engine_instance is None:
        _rule_engine_instance = ComponentRuleEngine(nutrition_registry)
    return _rule_engine_instance
