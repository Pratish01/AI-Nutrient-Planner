"""
3-Stage Hierarchical Food Recognition

Stage 1: Coarse Visual Traits (mandatory)
Stage 2: Mid-Level Food Type (narrow search)
Stage 3: Fine Dish Resolution (optional)

Principle: Ask smallest question first. Stop early if uncertain.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from PIL import Image
from enum import Enum
import torch
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1: COARSE VISUAL TRAITS
# =============================================================================

class VisualTrait(Enum):
    """Broad visual traits (multiple can apply)."""
    LIQUID_BASED = "liquid_based"
    DRY_FOOD = "dry_food"
    FRIED_FOOD = "fried_food"
    NON_FRIED_FOOD = "non_fried_food"
    HANDHELD_FOOD = "handheld_food"
    PLATED_MEAL = "plated_meal"
    RICE_BASED = "rice_based"
    BREAD_BASED = "bread_based"
    MIXED_SCENE = "mixed_scene"


TRAIT_PROMPTS: Dict[VisualTrait, str] = {
    VisualTrait.LIQUID_BASED: "food with visible liquid gravy or sauce",
    VisualTrait.DRY_FOOD: "dry food without gravy or sauce",
    VisualTrait.FRIED_FOOD: "deep fried crispy food",
    VisualTrait.NON_FRIED_FOOD: "non-fried steamed or baked food",
    VisualTrait.HANDHELD_FOOD: "handheld snack or finger food",
    VisualTrait.PLATED_MEAL: "full meal served on a plate",
    VisualTrait.RICE_BASED: "rice or grain based dish",
    VisualTrait.BREAD_BASED: "bread or flatbread based food",
    VisualTrait.MIXED_SCENE: "multiple small food items together",
}


# =============================================================================
# STAGE 2: MID-LEVEL FOOD TYPES
# =============================================================================

class FoodType(Enum):
    """Mid-level food categories."""
    STREET_FOOD = "street_food"
    FRIED_SNACK = "fried_snack"
    SANDWICH = "sandwich"
    DAL_LENTIL = "dal_lentil"
    CURRY = "curry"
    SOUP = "soup"
    RICE_DISH = "rice_dish"
    BREAD_MEAL = "bread_meal"
    SOUTH_INDIAN = "south_indian"
    DESSERT = "dessert"
    CONTINENTAL = "continental"


FOOD_TYPE_PROMPTS: Dict[FoodType, str] = {
    FoodType.STREET_FOOD: "indian street food like pani puri or chaat",
    FoodType.FRIED_SNACK: "fried snack like samosa or pakora",
    FoodType.SANDWICH: "sandwich or burger",
    FoodType.DAL_LENTIL: "dal or lentil curry in a bowl",
    FoodType.CURRY: "indian curry with thick gravy",
    FoodType.SOUP: "soup or broth",
    FoodType.RICE_DISH: "rice dish like biryani or pulao",
    FoodType.BREAD_MEAL: "indian bread like roti naan or paratha",
    FoodType.SOUTH_INDIAN: "south indian food like dosa or idli",
    FoodType.DESSERT: "indian dessert or sweet",
    FoodType.CONTINENTAL: "western food like pizza or pasta",
}


# Trait → Food Type mapping (which food types are consistent with which traits)
TRAIT_TO_FOOD_TYPES: Dict[VisualTrait, List[FoodType]] = {
    VisualTrait.LIQUID_BASED: [FoodType.DAL_LENTIL, FoodType.CURRY, FoodType.SOUP],
    VisualTrait.DRY_FOOD: [FoodType.FRIED_SNACK, FoodType.BREAD_MEAL, FoodType.SANDWICH],
    VisualTrait.FRIED_FOOD: [FoodType.FRIED_SNACK, FoodType.STREET_FOOD],
    VisualTrait.NON_FRIED_FOOD: [FoodType.CURRY, FoodType.DAL_LENTIL, FoodType.SOUTH_INDIAN],
    VisualTrait.HANDHELD_FOOD: [FoodType.FRIED_SNACK, FoodType.SANDWICH, FoodType.STREET_FOOD],
    VisualTrait.PLATED_MEAL: [FoodType.RICE_DISH, FoodType.CURRY, FoodType.BREAD_MEAL],
    VisualTrait.RICE_BASED: [FoodType.RICE_DISH, FoodType.SOUTH_INDIAN],
    VisualTrait.BREAD_BASED: [FoodType.BREAD_MEAL, FoodType.SANDWICH],
    VisualTrait.MIXED_SCENE: [FoodType.STREET_FOOD, FoodType.SOUTH_INDIAN],
}


# =============================================================================
# STAGE 3: DISH PROMPTS PER FOOD TYPE
# =============================================================================

# Max 6 dishes per food type
FOOD_TYPE_DISHES: Dict[FoodType, List[str]] = {
    FoodType.STREET_FOOD: [
        "Pani Puri",
        "Bhel Puri",
        "Dahi Puri",
        "Sev Puri",
        "Pav Bhaji",
        "Chaat",
    ],
    FoodType.FRIED_SNACK: [
        "Samosa",
        "Pakora",
        "Vada",
        "Kachori",
        "Bread Pakora",
        "Cutlet",
    ],
    FoodType.DAL_LENTIL: [
        "Dal Makhani",
        "Dal Tadka",
        "Sambar",
        "Rasam",
        "Mixed Dal",
        "Chana Dal",
    ],
    FoodType.CURRY: [
        "Paneer Butter Masala",
        "Butter Chicken",
        "Palak Paneer",
        "Kadhi",
        "Shahi Paneer",
        "Chicken Curry",
    ],
    FoodType.RICE_DISH: [
        "Biryani",
        "Pulao",
        "Jeera Rice",
        "Fried Rice",
        "Curd Rice",
        "Lemon Rice",
    ],
    FoodType.BREAD_MEAL: [
        "Naan",
        "Roti",
        "Paratha",
        "Poori",
        "Kulcha",
        "Bhatura",
    ],
    FoodType.SOUTH_INDIAN: [
        "Dosa",
        "Idli",
        "Vada",
        "Uttapam",
        "Appam",
        "Pongal",
    ],
    FoodType.DESSERT: [
        "Gulab Jamun",
        "Rasgulla",
        "Kheer",
        "Halwa",
        "Jalebi",
        "Barfi",
    ],
    FoodType.SANDWICH: [
        "Veg Sandwich",
        "Cheese Sandwich",
        "Burger",
        "Club Sandwich",
    ],
    FoodType.CONTINENTAL: [
        "Pizza",
        "Pasta",
        "French Fries",
        "Salad",
    ],
    FoodType.SOUP: [
        "Tomato Soup",
        "Manchow Soup",
        "Hot and Sour Soup",
    ],
}


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class HierarchicalResult:
    """Result from 3-stage hierarchical recognition."""
    stage1_traits: List[str]
    stage2_food_type: str
    resolved_food: str
    resolution_stage: str  # "stage3" | "stage2" | "unknown"
    confidence: float


# =============================================================================
# HIERARCHICAL CLASSIFIER
# =============================================================================

class HierarchicalClassifier:
    """
    3-Stage Hierarchical Food Recognition.
    
    Stage 1: Coarse traits → Stage 2: Food type → Stage 3: Dish (optional)
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        trait_threshold: float = 0.30,
        type_threshold: float = 0.25,
        dish_threshold: float = 0.25,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.trait_threshold = trait_threshold
        self.type_threshold = type_threshold
        self.dish_threshold = dish_threshold
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        
        # Pre-computed embeddings
        self._trait_embeddings = None
        self._type_embeddings = None
        self._dish_embeddings: Dict[FoodType, torch.Tensor] = {}
    
    def _ensure_initialized(self):
        """Lazy initialization."""
        if self.model is not None:
            return
        
        import open_clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HierarchicalClassifier] Loading on {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute text embeddings for Stage 1 and 2."""
        # Stage 1: Traits
        trait_prompts = [TRAIT_PROMPTS[t] for t in VisualTrait]
        tokens = self.tokenizer(trait_prompts).to(self.device)
        with torch.no_grad():
            self._trait_embeddings = self.model.encode_text(tokens)
            self._trait_embeddings = self._trait_embeddings / self._trait_embeddings.norm(dim=-1, keepdim=True)
        
        # Stage 2: Food Types
        type_prompts = [FOOD_TYPE_PROMPTS[t] for t in FoodType]
        tokens = self.tokenizer(type_prompts).to(self.device)
        with torch.no_grad():
            self._type_embeddings = self.model.encode_text(tokens)
            self._type_embeddings = self._type_embeddings / self._type_embeddings.norm(dim=-1, keepdim=True)
        
        logger.info(f"[HierarchicalClassifier] Pre-computed {len(trait_prompts)} traits, {len(type_prompts)} types")
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image once, reuse for all stages."""
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding
    
    def _stage1_traits(self, image_embedding: torch.Tensor) -> Tuple[List[VisualTrait], Dict[VisualTrait, float]]:
        """Stage 1: Detect coarse visual traits."""
        similarities = (image_embedding @ self._trait_embeddings.T).squeeze(0)
        probs = torch.sigmoid(similarities * 5)  # Binary detection
        
        detected_traits = []
        confidences = {}
        
        for i, trait in enumerate(VisualTrait):
            conf = float(probs[i].cpu())
            confidences[trait] = conf
            if conf >= self.trait_threshold:
                detected_traits.append(trait)
        
        logger.info(f"[Stage1] Traits: {[t.value for t in detected_traits]}")
        return detected_traits, confidences
    
    def _stage2_food_type(
        self,
        image_embedding: torch.Tensor,
        traits: List[VisualTrait],
    ) -> Tuple[Optional[FoodType], float]:
        """Stage 2: Narrow to food type based on traits."""
        # Get candidate food types from traits
        candidates: Set[FoodType] = set()
        for trait in traits:
            if trait in TRAIT_TO_FOOD_TYPES:
                candidates.update(TRAIT_TO_FOOD_TYPES[trait])
        
        if not candidates:
            # No trait mapping, use all types
            candidates = set(FoodType)
        
        # Limit to max 8 types
        candidate_list = list(candidates)[:8]
        
        logger.info(f"[Stage2] Evaluating {len(candidate_list)} food types")
        
        # Compute similarities for candidates only
        candidate_indices = [list(FoodType).index(c) for c in candidate_list]
        candidate_embeddings = self._type_embeddings[candidate_indices]
        
        similarities = (image_embedding @ candidate_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0)
        
        top_idx = torch.argmax(probs).item()
        top_conf = float(probs[top_idx].cpu())
        top_type = candidate_list[top_idx]
        
        if top_conf < self.type_threshold:
            logger.info(f"[Stage2] Low confidence ({top_conf:.2f}), stopping")
            return None, top_conf
        
        logger.info(f"[Stage2] Food type: {top_type.value} ({top_conf:.2f})")
        return top_type, top_conf
    
    def _stage3_dish(
        self,
        image_embedding: torch.Tensor,
        food_type: FoodType,
        nutrition_registry=None,
    ) -> Tuple[Optional[str], float]:
        """Stage 3: Fine dish resolution (limited)."""
        dishes = FOOD_TYPE_DISHES.get(food_type, [])
        if not dishes:
            return None, 0.0
        
        # Limit to max 6 dishes
        dishes = dishes[:6]
        
        logger.info(f"[Stage3] Evaluating {len(dishes)} dishes for {food_type.value}")
        
        # Compute dish embeddings (cache for efficiency)
        if food_type not in self._dish_embeddings:
            prompts = [f"a photo of {dish}" for dish in FOOD_TYPE_DISHES.get(food_type, [])]
            if prompts:
                tokens = self.tokenizer(prompts).to(self.device)
                with torch.no_grad():
                    embeddings = self.model.encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                self._dish_embeddings[food_type] = embeddings
        
        dish_embeddings = self._dish_embeddings.get(food_type)
        if dish_embeddings is None:
            return None, 0.0
        
        similarities = (image_embedding @ dish_embeddings.T).squeeze(0)
        probs = torch.softmax(similarities * 100, dim=0)
        
        top_idx = torch.argmax(probs).item()
        top_conf = float(probs[top_idx].cpu())
        top_dish = dishes[top_idx]
        
        # Check threshold
        if top_conf < self.dish_threshold:
            logger.info(f"[Stage3] Low confidence ({top_conf:.2f}), fallback to food type")
            return None, top_conf
        
        # Verify in database (if available)
        if nutrition_registry:
            item = nutrition_registry.get_by_name(top_dish)
            if not item:
                logger.info(f"[Stage3] '{top_dish}' not in database, fallback")
                return None, top_conf
        
        logger.info(f"[Stage3] Dish: {top_dish} ({top_conf:.2f})")
        return top_dish, top_conf
    
    def classify(
        self,
        image: Image.Image,
        nutrition_registry=None,
    ) -> HierarchicalResult:
        """
        Run 3-stage hierarchical classification.
        
        Returns HierarchicalResult with resolution at highest confident stage.
        """
        self._ensure_initialized()
        
        # Single image encoding (reused for all stages)
        image_embedding = self._encode_image(image)
        
        # ===== STAGE 1: Coarse Traits =====
        traits, trait_confs = self._stage1_traits(image_embedding)
        
        if not traits:
            # No traits detected - return unknown
            return HierarchicalResult(
                stage1_traits=[],
                stage2_food_type="unknown",
                resolved_food="unknown",
                resolution_stage="unknown",
                confidence=0.0,
            )
        
        # ===== STAGE 2: Food Type =====
        food_type, type_conf = self._stage2_food_type(image_embedding, traits)
        
        if food_type is None:
            # Low confidence - return traits only
            return HierarchicalResult(
                stage1_traits=[t.value for t in traits],
                stage2_food_type="unknown",
                resolved_food="unknown",
                resolution_stage="unknown",
                confidence=type_conf,
            )
        
        # ===== STAGE 3: Dish Resolution (optional) =====
        dish, dish_conf = self._stage3_dish(image_embedding, food_type, nutrition_registry)
        
        if dish:
            return HierarchicalResult(
                stage1_traits=[t.value for t in traits],
                stage2_food_type=food_type.value,
                resolved_food=dish,
                resolution_stage="stage3",
                confidence=dish_conf,
            )
        
        # Fallback to food type
        return HierarchicalResult(
            stage1_traits=[t.value for t in traits],
            stage2_food_type=food_type.value,
            resolved_food=food_type.value,
            resolution_stage="stage2",
            confidence=type_conf,
        )


# =============================================================================
# SINGLETON
# =============================================================================

_classifier_instance: Optional[HierarchicalClassifier] = None

def get_hierarchical_classifier() -> HierarchicalClassifier:
    """Get or create the hierarchical classifier singleton."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = HierarchicalClassifier()
    return _classifier_instance
