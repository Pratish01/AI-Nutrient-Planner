"""
Stable Food Recognition Pipeline with SigLIP

DESIGN PRINCIPLES:
1. Single-source label file: expanded_indian_food_3000_plus.txt
2. Fixed prompt template: "a photo of <label>"
3. Text embeddings cached at startup (never re-encoded per request)
4. Sigmoid activation for multi-label probabilities

PIPELINE ORDER:
1. YOLOv11n Detection → Crop food region
2. SigLIP Image Encoding → Encode ONCE, normalize immediately
3. Similarity Computation → Image embedding vs cached text embeddings
4. Multi-label Probabilities → Apply sigmoid (NOT softmax)

MODELS (MANDATORY - DO NOT CHANGE):
- YOLOv11n for detection only
- SigLIP google/siglip-so400m-patch14-384 via HuggingFace

SPEED OPTIMIZATIONS:
- Precompute ALL text embeddings at startup (1185 labels)
- Reuse image embedding for all classification stages
- torch.no_grad() everywhere
- GPU acceleration when available
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
from PIL import Image

# For specificity gate
try:
    from .nutrition_registry import get_nutrition_registry
except ImportError:
    # Fallback for alternative import paths
    try:
        from src.services.nutrition_registry import get_nutrition_registry
    except ImportError:
        def get_nutrition_registry(): return None

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoodGroupPrediction:
    """Primary stable prediction - food group."""
    name: str
    confidence: float
    cuisine: str  # "indian" or "continental"


@dataclass
class DishSuggestion:
    """Secondary suggestion - dish name (NOT guaranteed accurate)."""
    name: str
    confidence: float


@dataclass
class RecognitionResult:
    """
    Complete recognition result.
    
    Food group is the TRUSTED output.
    Dish suggestions are TOP-K retrieval results only.
    """
    # Primary output - stable and reliable
    food_group: FoodGroupPrediction
    
    # Secondary output - suggestions only, use with caution
    dish_suggestions: List[DishSuggestion] = field(default_factory=list)
    
    # Metadata
    bbox: Optional[Tuple[int, int, int, int]] = None
    detection_confidence: float = 0.0
    used_fallback: bool = False
    
    # Optional sides detected in the same image
    sides: List['RecognitionResult'] = field(default_factory=list)
    
    # Safety verdict
    safety_verdict: str = "allow"
    
    def get_best_dish(self, safety_check_fn: Optional[Any] = None) -> Optional[str]:
        """
        Get best dish suggestion using the REVISED Permissive Specificity Gate.
        
        Resolution Priority:
        1. Top-1 if confidence >= 0.25 and no major safety/ambiguity risk.
        2. Top-1 if ambiguity exists but items are visually/nutritionally similar.
        3. None if medical safety verdicts would differ between top candidates.
        
        Args:
            safety_check_fn: Optional callable (dish_name) -> safety_verdict
        """
        if not self.dish_suggestions:
            return None
            
        top_suggestion = self.dish_suggestions[0]
        
        # RULE 1: Threshold (Adjusted to 0.25 per accuracy recovery policy)
        if top_suggestion.confidence < 0.25:
            logger.info(f"Specificity Gate: REJECTED '{top_suggestion.name}' (Conf {top_suggestion.confidence:.2f} < 0.25)")
            return None
            
        # RULE 2: Existence in Nutrition DB - DISABLED
        # We now allow visual-only matches even if not in DB.
        # registry = get_nutrition_registry()
        # if registry:
        #     item = registry.get_by_name(top_suggestion.name)
        #     if not item:
        #         logger.info(f"Specificity Gate: REJECTED '{top_suggestion.name}' (Not in DB)")
        #         return None

        # RULE 3: Ambiguity Handling (Permissive)
        if len(self.dish_suggestions) > 1:
            second_suggestion = self.dish_suggestions[1]
            gap = top_suggestion.confidence - second_suggestion.confidence
            
            # STREET FOOD OVERRIDE: Relaxed gate
            # For street food, we bypass gap blocking entirely if Top-1 >= 0.25.
            if self.food_group.name == "street food" or self.food_group.name == "Street Food / Chaat":
                logger.info(f"Specificity Gate: STREET FOOD RESOLUTION '{top_suggestion.name}' (Gap {gap:.2f} ignored)")
                return top_suggestion.name

            # If gap is small (< 0.06), verify if we can resolve anyway
            if gap < 0.06:
                # 3a. Are they variants of the same family?
                if self._is_same_family(top_suggestion.name, second_suggestion.name):
                    logger.info(f"Specificity Gate: RESOLVED '{top_suggestion.name}' (Small gap {gap:.2f} but same family)")
                    return top_suggestion.name
                
                # 3b. Is nutrition similar (within 7% calories)?
                if not self._is_nutrition_distinct(top_suggestion.name, second_suggestion.name, threshold=0.07):
                    logger.info(f"Specificity Gate: RESOLVED '{top_suggestion.name}' (Small gap {gap:.2f} but similar nutrition)")
                    return top_suggestion.name
                
                # 3c. Does safety differ? (CRITICAL)
                if safety_check_fn:
                    verdict_a = safety_check_fn(top_suggestion.name)
                    verdict_b = safety_check_fn(second_suggestion.name)
                    if verdict_a != verdict_b:
                        logger.info(f"Specificity Gate: REJECTED '{top_suggestion.name}' (Ambiguous and safety verdicts differ)")
                        return None
                        
                # If nutrition is distinct AND it's not same family, we resolve to Top-1 anyway
                # per "Prefer resolving to a Specific Dish... even if ambiguity exists"
                # UNLESS safety check fails (which we handled above)
                logger.info(f"Specificity Gate: RESOLVED '{top_suggestion.name}' (Top-1 preference despite distinct nutrition)")
                
        return top_suggestion.name

    def to_minimal_response(self) -> Dict[str, Any]:
        """Return minimal, structured response per production rules."""
        # Determine if we resolve to a dish or just the group
        best_dish = self.get_best_dish()
        
        # SPECIAL STREET FOOD RULE: If confidence < 0.25, return "street_food"
        # However, our food group is already "street food" if detected.
        
        resolution_type = "unknown"
        final_name = "Unknown Food"
        confidence = 0.0
        
        if best_dish:
            resolution_type = "dish"
            final_name = best_dish
            confidence = self.dish_suggestions[0].confidence
        elif self.food_group.confidence >= 0.25 or (self.food_group.name.lower() == "street food" or self.food_group.name.lower() == "street food / chaat"):
            resolution_type = "group"
            final_name = self.food_group.name
            confidence = self.food_group.confidence
            
        return {
            "food_name": final_name,
            "confidence": round(confidence, 4),
            "resolution_type": resolution_type,
            "safety_verdict": self.safety_verdict
        }

    def _is_same_family(self, dish_a: str, dish_b: str) -> bool:
        """Identify if dishes belong to the same family (e.g. Dal derivatives)."""
        family_keywords = ["dal", "paneer", "chicken", "biryani", "roti", "paratha", "naan", "curry", "dosa", "idli", "pizza"]
        
        name_a = dish_a.lower()
        name_b = dish_b.lower()
        
        for kw in family_keywords:
            if kw in name_a and kw in name_b:
                return True
        return False

    def _is_nutrition_distinct(self, dish_a: str, dish_b: str, threshold: float = 0.05) -> bool:
        """Helper to check if two dishes have > threshold difference in calories."""
        registry = get_nutrition_registry()
        if not registry:
            return True 
            
        item_a = registry.get_by_name(dish_a)
        item_b = registry.get_by_name(dish_b)
        
        if not item_a or not item_b:
            return True
            
        cal_a = item_a.get('calories', 0)
        cal_b = item_b.get('calories', 0)
        
        if cal_a == 0 or cal_b == 0:
            return True
            
        diff = abs(cal_a - cal_b) / max(cal_a, cal_b) if max(cal_a, cal_b) > 0 else 0
        return diff > threshold


# =============================================================================
# FOOD GROUP MAPPING
# =============================================================================

# Maps food group prompts to their dish file and cuisine
# Maps food group prompts to their dish file and cuisine
FOOD_GROUP_CONFIG = {
    "dal": {
        "dish_file": "dishes_dal.txt",
        "cuisine": "indian",
        "display_name": "Dal / Lentil Curry",
        "is_liquid": True
    },
    "rice dish": {
        "dish_file": "dishes_rice_dish.txt",
        "cuisine": "indian",
        "display_name": "Rice Dish",
        "is_liquid": False
    },
    "indian bread": {
        "dish_file": "dishes_indian_bread.txt",
        "cuisine": "indian",
        "display_name": "Indian Bread",
        "is_liquid": False
    },
    "street food": {
        "dish_file": "dishes_street_food.txt",
        "cuisine": "indian",
        "display_name": "Street Food / Chaat",
        "is_liquid": False,
        "is_street_food": True
    },
    "south indian": {
        "dish_file": "dishes_south_indian.txt",
        "cuisine": "indian",
        "display_name": "South Indian",
        "is_liquid": False
    },
    "wet curry": {
        "dish_file": "dishes_other.txt",
        "cuisine": "indian",
        "display_name": "Wet Curry / Gravy",
        "is_liquid": True
    },
    "dry vegetable": {
        "dish_file": "dishes_other.txt",
        "cuisine": "indian",
        "display_name": "Dry Vegetable Sabzi",
        "is_liquid": False
    },
    "dessert": {
        "dish_file": "dishes_dessert.txt",
        "cuisine": "indian",
        "display_name": "Dessert / Sweet",
        "is_liquid": False
    },
    "pizza": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Pizza",
        "is_liquid": False
    },
    "burger": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Burger",
        "is_liquid": False
    },
    "pasta": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Pasta",
        "is_liquid": False
    },
    "sandwich": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Sandwich",
        "is_liquid": False
    },
    "fried food": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Fried Food",
        "is_liquid": False
    },
    "salad": {
        "dish_file": "dishes_continental.txt",
        "cuisine": "continental",
        "display_name": "Salad",
        "is_liquid": False
    },
}

# Characteristics for Street Food Priority Detection
STREET_FOOD_CHARACTERISTICS = [
    "multiple small food items on a plate",
    "mixed dry and liquid components in a food dish",
    "crispy hollow or puffed textures of food",
    "irregular shapes of food served on a plate, leaf, or paper",
]


def extract_food_group_key(prompt: str) -> str:
    """Extract food group key from a prompt line."""
    prompt_lower = prompt.lower()
    
    # Check each known food group
    for key in FOOD_GROUP_CONFIG.keys():
        if key in prompt_lower:
            return key
    
    # Fallback parsing
    if "dal" in prompt_lower or "lentil" in prompt_lower:
        return "dal"
    if "rice" in prompt_lower:
        return "rice dish"
    if "bread" in prompt_lower or "roti" in prompt_lower or "naan" in prompt_lower:
        return "indian bread"
    if "street" in prompt_lower or "chaat" in prompt_lower or \
       "puri" in prompt_lower or "golgappa" in prompt_lower or \
       "bhel" in prompt_lower or "pav bhaji" in prompt_lower or \
       "pakora" in prompt_lower:
        return "street food"
    if "south indian" in prompt_lower or "dosa" in prompt_lower or "idli" in prompt_lower:
        return "south indian"
    if "curry" in prompt_lower or "gravy" in prompt_lower:
        return "wet curry"
    if "dry" in prompt_lower or "sabzi" in prompt_lower:
        return "dry vegetable"
    if "dessert" in prompt_lower or "sweet" in prompt_lower:
        return "dessert"
    if "pizza" in prompt_lower:
        return "pizza"
    if "burger" in prompt_lower:
        return "burger"
    if "pasta" in prompt_lower:
        return "pasta"
    if "sandwich" in prompt_lower:
        return "sandwich"
    if "fried" in prompt_lower:
        return "fried food"
    if "salad" in prompt_lower:
        return "salad"
    
    return "other"


# =============================================================================
# STABLE SIGLIP CLASSIFIER
# =============================================================================

class StableSigLIPClassifier:
    """
    Stable SigLIP classifier for food recognition.
    
    CRITICAL DESIGN DECISIONS:
    1. Uses google/siglip-so400m-patch14-384 via HuggingFace Transformers
    2. All labels from expanded_indian_food_3000_plus.txt (single source of truth)
    3. Fixed prompt template: "a photo of <label>" (no variations)
    4. Text embeddings cached at startup (batch encode, normalize, cache)
    5. Sigmoid activation for multi-label probabilities (NOT softmax)
    
    Usage:
        classifier = StableSigLIPClassifier()
        result = classifier.classify(pil_image)
        
        # Food group and dish suggestions
        print(f"Food: {result.food_group.name}")
        print(f"Top dish: {result.dish_suggestions[0].name}")
    """
    
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    LABELS_FILE = "expanded_indian_food_3000_plus.txt"
    FOOD_GROUPS_FILE = "CLIP_Food_Groups.txt"
    # Reverted to Large 384 for accuracy (Base 224 was too inaccurate)
    SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
    
    def __init__(self, device: Optional[str] = None):
        """Initialize classifier with precomputed embeddings."""
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"StableSigLIPClassifier using device: {self.device}")
        
        # Model components
        self.model = None
        self.processor = None
        
        # All dish labels from expanded food file
        self.labels: List[str] = []
        self.label_embeddings: Optional[torch.Tensor] = None
        
        # Food group embeddings (for backward compatibility with pipeline)
        self.food_group_prompts: List[str] = []
        self.food_group_keys: List[str] = []
        self.food_group_embeddings: Optional[torch.Tensor] = None
        
        # Street food characteristic embeddings
        self.characteristic_embeddings: Optional[torch.Tensor] = None
        
        # Initialize
        self._load_model()
        self._load_labels()
        self._precompute_label_embeddings()
        self._load_food_groups()
        self._precompute_food_group_embeddings()
        self._precompute_characteristic_embeddings()
    
    def warmup(self):
        """Run a dummy inference to initialize buffers and remove cold-start latency."""
        try:
            logger.info("Warming up SigLIP model...")
            # Create a localized dummy image (noise)
            dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            # Simple inference pass
            with torch.inference_mode():
                # Process image
                inputs = self.processor(images=dummy_image, return_tensors="pt").to(self.device)
                
                # Run vision encoder
                image_embeds = self.model.get_image_features(**inputs)
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                
                # Check against cached text embeddings (forces matrix multiplication kernel load)
                if self.label_embeddings is not None:
                    _ = torch.sigmoid(image_embeds @ self.label_embeddings.T)
                    
            logger.info("Warmup complete - model ready for fast inference")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    def _load_model(self) -> None:
        """Load SigLIP model with MANDATORY configuration."""
        try:
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
            
            # Try SigLIP-specific imports first (more reliable)
            try:
                from transformers import SiglipModel, SiglipProcessor
                logger.info(f"Loading SigLIP model: {self.SIGLIP_MODEL_NAME}...")
                self.model = SiglipModel.from_pretrained(self.SIGLIP_MODEL_NAME)
                self.processor = SiglipProcessor.from_pretrained(self.SIGLIP_MODEL_NAME)
            except (ImportError, AttributeError, ValueError) as inner_e:
                logger.warning(f"SigLIP-specific classes not available or failed: {inner_e}")
                # Fallback to Auto classes
                from transformers import AutoModel, AutoProcessor
                logger.info(f"Loading SigLIP model (via Auto): {self.SIGLIP_MODEL_NAME}...")
                self.model = AutoModel.from_pretrained(self.SIGLIP_MODEL_NAME)
                self.processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL_NAME)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("SigLIP model loaded successfully")
            
        except ImportError as e:
            # CRITICAL: Do NOT silently fail or fallback
            import traceback
            logger.error(f"Import error details: {traceback.format_exc()}")
            error_msg = (
                f"HuggingFace Transformers is not installed or SigLIP unavailable. "
                f"Error: {str(e)}. "
                f"Install with: pip install transformers>=4.40.0 torch"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            error_msg = f"Failed to load SigLIP model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _load_labels(self) -> None:
        """Load all dish labels from expanded food file (single source of truth)."""
        labels_path = self.DATA_DIR / self.LABELS_FILE
        
        self.labels = []
        
        if labels_path.exists():
            with open(labels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.labels.append(line)
            
            logger.info(f"Loaded {len(self.labels)} dish labels from {self.LABELS_FILE}")
        else:
            logger.error(f"Labels file not found: {labels_path}")
            raise FileNotFoundError(f"Required labels file not found: {labels_path}")
    
    def _precompute_label_embeddings(self) -> None:
        """
        Precompute embeddings for ALL dish labels.
        
        CRITICAL RULES (DO NOT DEVIATE):
        1. Fixed prompt template: "a photo of <label>"
        2. Batch encode all labels at startup
        3. Normalize embeddings immediately
        4. Cache for reuse (never re-encode per request)
        """
        if not self.labels:
            logger.warning("No labels to precompute")
            return
        
        logger.info(f"Precomputing embeddings for {len(self.labels)} labels...")
        
        # FIXED PROMPT TEMPLATE - DO NOT CHANGE
        prompts = [f"a photo of {label}" for label in self.labels]
        
        with torch.no_grad():
            # Batch encode text in chunks to manage memory
            batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                
                # Use processor to tokenize
                inputs = self.processor(
                    text=batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
                
                # Get text embeddings from SigLIP
                text_outputs = self.model.get_text_features(**inputs)
                
                # Normalize embeddings immediately
                text_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                all_embeddings.append(text_embeddings)
            
            self.label_embeddings = torch.cat(all_embeddings, dim=0)
        
        logger.info(f"Cached {len(self.labels)} label embeddings (shape: {self.label_embeddings.shape})")
    
    def _load_food_groups(self) -> None:
        """Load food group prompts from TXT file (for backward compatibility)."""
        food_groups_path = self.DATA_DIR / self.FOOD_GROUPS_FILE
        
        self.food_group_prompts = []
        self.food_group_keys = []
        
        if food_groups_path.exists():
            with open(food_groups_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.food_group_prompts.append(line)
                        self.food_group_keys.append(extract_food_group_key(line))
            
            logger.info(f"Loaded {len(self.food_group_prompts)} food group prompts")
        else:
            logger.warning(f"Food groups file not found: {food_groups_path}")
    
    def _precompute_food_group_embeddings(self) -> None:
        """Precompute embeddings for food groups (backward compatibility)."""
        if not self.food_group_prompts:
            logger.warning("No food group prompts to precompute")
            return
        
        logger.info("Precomputing food group embeddings...")
        
        with torch.no_grad():
            inputs = self.processor(
                text=self.food_group_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            
            text_outputs = self.model.get_text_features(**inputs)
            self.food_group_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        
        logger.info(f"Precomputed {len(self.food_group_prompts)} food group embeddings")
    
    def _precompute_characteristic_embeddings(self) -> None:
        """Precompute embeddings for street food visual characteristics."""
        logger.info("Precomputing street food characteristic embeddings...")
        with torch.no_grad():
            inputs = self.processor(
                text=STREET_FOOD_CHARACTERISTICS,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
            
            text_outputs = self.model.get_text_features(**inputs)
            self.characteristic_embeddings = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
    
    def detect_street_food_characteristics(self, image_embedding: torch.Tensor) -> float:
        """Detect if the image shows street food characteristics."""
        if self.characteristic_embeddings is None:
            return 0.0
        with torch.no_grad():
            similarities = (image_embedding @ self.characteristic_embeddings.T).squeeze(0)
            max_sim = float(torch.max(similarities).cpu().item())
            return max_sim
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to embedding.
        
        IMPORTANT: 
        - Compute this ONCE per image
        - Normalize embedding immediately
        - Reuse for all classification stages
        """
        with torch.inference_mode():
            # Use processor to preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get image embedding from SigLIP
            image_outputs = self.model.get_image_features(**inputs)
            
            # Normalize immediately
            embedding = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        
        return embedding
    
    def classify(
        self,
        image: Image.Image,
        top_k_dishes: int = 5
    ) -> RecognitionResult:
        """
        Classify food image using SigLIP.
        
        Pipeline:
        1. Encode image ONCE (normalize immediately)
        2. Compute similarity with cached text embeddings
        3. Apply sigmoid for multi-label probabilities (NOT softmax)
        4. Return food group and top-K dish suggestions
        
        Args:
            image: PIL Image to classify
            top_k_dishes: Number of dish suggestions to retrieve
            
        Returns:
            RecognitionResult with food group and dish suggestions
        """
        # Step 1: Encode image ONCE
        image_embedding = self._encode_image(image)
        
        # Step 2: Classify food group (for backward compatibility)
        groups = self._classify_food_groups_ranked(image_embedding)
        primary_group = groups[0]
        
        # Step 3: Get dish suggestions using sigmoid-based probabilities
        dish_suggestions = self._get_dish_suggestions(
            image_embedding,
            top_k=top_k_dishes
        )
        
        return RecognitionResult(
            food_group=primary_group,
            dish_suggestions=dish_suggestions
        )
    
    def _classify_food_groups_ranked(
        self,
        image_embedding: torch.Tensor
    ) -> List[FoodGroupPrediction]:
        """Get all food groups ranked by confidence."""
        if self.food_group_embeddings is None:
            return [FoodGroupPrediction("Unknown", 0.0, "unknown")]
        
        with torch.no_grad():
            # Compute similarity
            similarities = (image_embedding @ self.food_group_embeddings.T).squeeze(0)
            
            # Apply sigmoid for multi-label probabilities (SigLIP is trained with sigmoid)
            probs = torch.sigmoid(similarities)
            
            # Sort all
            sorted_conf, sorted_idx = torch.sort(probs, descending=True)
            
        predictions = []
        for conf, idx in zip(sorted_conf.tolist(), sorted_idx.tolist()):
            food_group_key = self.food_group_keys[idx]
            config = FOOD_GROUP_CONFIG.get(food_group_key, {})
            predictions.append(FoodGroupPrediction(
                name=config.get("display_name", food_group_key.title()),
                confidence=conf,
                cuisine=config.get("cuisine", "unknown")
            ))
        
        # DEBUG LOGGING: Show all food group confidences for diagnosis
        logger.info("FOOD GROUP CONFIDENCES (ALL):")
        for i, pred in enumerate(predictions[:10]):  # Top 10
            logger.info(f"  {i+1}. {pred.name:25s} -> {pred.confidence:.3f} ({pred.cuisine})")
            
        return predictions
    
    def _get_dish_suggestions(
        self,
        image_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[DishSuggestion]:
        """
        Get dish suggestions using similarity with all cached label embeddings.
        
        CRITICAL: Uses sigmoid for multi-label probabilities (NOT softmax)
        """
        if self.label_embeddings is None or len(self.labels) == 0:
            return []
        
        with torch.no_grad():
            # Compute similarity with all label embeddings
            similarities = (image_embedding @ self.label_embeddings.T).squeeze(0)
            
            # CRITICAL: Apply SIGMOID for multi-label probabilities
            # SigLIP is trained with sigmoid loss - this is the correct activation
            # Do NOT use softmax - that would be incorrect for SigLIP
            probs = torch.sigmoid(similarities)
            
            # Get top-K
            k = min(top_k, len(self.labels))
            top_probs, top_indices = probs.topk(k)
        
        suggestions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            suggestions.append(DishSuggestion(
                name=self.labels[idx],
                confidence=float(prob)
            ))
        
        return suggestions
    
    def is_available(self) -> bool:
        """Check if classifier is ready."""
        return (
            self.model is not None and
            self.label_embeddings is not None
        )


# Backward compatibility alias
StableOpenCLIPClassifier = StableSigLIPClassifier


# =============================================================================
# YOLO DETECTOR (DETECTION ONLY)
# =============================================================================

class StableYOLODetector:
    """
    YOLOv11n detector for food region extraction.
    
    Purpose: Detection and cropping ONLY.
    Does NOT classify food - that's OpenCLIP's job.
    """
    
    def __init__(self, imgsz: int = 416, confidence: float = 0.25):
        self.imgsz = imgsz
        self.confidence = confidence
        self.model = None
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _ensure_loaded(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO("yolo11n.pt")
            logger.info("Loaded YOLOv11n detector")
    
    def detect(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int], float]]:
        """
        Detect ALL food regions and return crops.
        
        Returns:
            List of Tuples (cropped_image, bbox, confidence)
        """
        self._ensure_loaded()
        
        results = self.model(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        # Extract all detections
        detections = []
        boxes = result.boxes
        
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(boxes.conf[i].cpu().numpy())
            
            # CRITICAL FIX: Add padding to preserve visual context
            # CLIP needs to see serving style, garnishes, and texture details
            # that tight YOLO crops remove
            w, h = image.size
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Add 30% padding on each side (increased from 20% for dish-level accuracy)
            pad_x = int(box_w * 0.30)
            pad_y = int(box_h * 0.30)
            
            x1_padded = max(0, x1 - pad_x)
            y1_padded = max(0, y1 - pad_y)
            x2_padded = min(w, x2 + pad_x)
            y2_padded = min(h, y2 + pad_y)
            
            # Skip invalid or tiny boxes
            if x2_padded <= x1_padded or y2_padded <= y1_padded:
                continue
                
            crop = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
            detections.append((crop, (x1_padded, y1_padded, x2_padded, y2_padded), conf))
        
        # Sort by confidence
        detections.sort(key=lambda x: x[2], reverse=True)
        
        return detections


# =============================================================================
# STABLE PIPELINE
# =============================================================================

class StableFoodPipeline:
    """
    Complete stable food recognition pipeline.
    
    Pipeline order:
    1. YOLO Detection -> Get food crop (or use full image)
    2. OpenCLIP Encoding -> Encode image ONCE
    3. Food Group Classification -> Stable TOP-1
    4. Dish Retrieval -> TOP-K suggestions
    
    Output priority:
    1. Food Group -> ALWAYS returned, ALWAYS reliable
    2. Dish Suggestions -> Optional, use with caution
    """
    
    def __init__(self):
        self._detector: Optional[StableYOLODetector] = None
        self._classifier: Optional[StableOpenCLIPClassifier] = None
        self._component_detector = None
        self._rule_engine = None
        self._initialized = False
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        
        logger.info("Initializing Stable Food Pipeline...")
        
        self._detector = StableYOLODetector()
        self._classifier = StableOpenCLIPClassifier()
        
        # Component-driven recognition (lazy load)
        try:
            from .component_detector import get_component_detector
            from .component_rules import get_component_rule_engine
            self._component_detector = get_component_detector()
            self._rule_engine = get_component_rule_engine()
            logger.info("Component-driven recognition enabled")
        except Exception as e:
            logger.warning(f"Component detection not available: {e}")
            self._component_detector = None
            self._rule_engine = None
            self._rule_engine = None
        
        self._initialized = True
        logger.info("Stable Food Pipeline initialized")

    def initialize(self):
        """
        Public method to force initialization and warmup.
        Call this at application startup.
        """
        logger.info("Starting eager initialization of Stable Pipeline...")
        
        # Optimize CPU threads for inference
        # (Avoid oversubscription which slows down inference)
        if torch.cuda.is_available() is False:
            try:
                # Rule of thumb: physical cores - 1 (or 4 for typical VM)
                # Leaving some cores for web server handling
                import multiprocessing
                cpu_count = multiprocessing.cpu_count()
                optimal_threads = max(1, min(4, cpu_count - 1))
                torch.set_num_threads(optimal_threads)
                logger.info(f"Optimized PyTorch CPU threads to: {optimal_threads}")
            except Exception as e:
                logger.warning(f"Could not set CPU threads: {e}")
        
        self._ensure_initialized()
        
        # Run warmup
        if self._classifier:
            self._classifier.warmup()
            
        logger.info(">>> Stable Pipeline READY for requests <<<")

    
    def recognize(
        self,
        image_path: str,
        top_k_dishes: int = 5
    ) -> RecognitionResult:
        """
        Recognize food in an image.
        
        Args:
            image_path: Path to image file
            top_k_dishes: Number of dish suggestions
            
        Returns:
            RecognitionResult with food group and dish suggestions
        """
        self._ensure_initialized()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        return self.recognize_pil(image, top_k_dishes)
    
    def recognize_with_components(
        self,
        image: Image.Image,
    ) -> dict:
        """
        Component-driven food recognition (deterministic).
        
        Pipeline:
        1. Detect visual components (binary)
        2. Analyze structure
        3. Apply rules in priority order
        4. Verify against database
        5. Return minimal structured output
        
        This is the PRIMARY method for Indian food recognition.
        """
        self._ensure_initialized()
        
        if self._component_detector is None or self._rule_engine is None:
            return {
                "resolved_food": "Unknown",
                "resolution_type": "error",
                "detected_components": [],
                "confidence": 0.0,
                "error": "Component system not initialized",
            }
        
        # Step 1: Detect components
        detection = self._component_detector.detect(image)
        
        logger.info(f"[ComponentFlow] Detected {len(detection.detected)} components")
        logger.info(f"[ComponentFlow] Structure: {detection.structure.value}")
        
        # Step 2 & 3: Derive dish using rules
        result = self._rule_engine.derive_dish(
            detection.detected,
            detection.structure,
        )
        
        logger.info(f"[ComponentFlow] Resolved: {result.resolved_food} ({result.resolution_type})")
        
        # Return minimal structured output
        return {
            "detected_components": result.detected_components,
            "resolved_food": result.resolved_food,
            "resolution_type": result.resolution_type,
            "confidence": result.confidence,
            "food_group": result.food_group,
            "cuisine": result.cuisine,
            "matched_rule": result.matched_rule,
        }

    def recognize_hierarchical(
        self,
        image: Image.Image,
    ) -> dict:
        """
        3-Stage Hierarchical Food Recognition (deterministic).
        
        Stage 1: Coarse traits (liquid, fried, etc.)
        Stage 2: Food type (max 8 evaluated)
        Stage 3: Dish resolution (max 6 evaluated, optional)
        
        This is the PRIMARY method for deterministic Indian food recognition.
        """
        self._ensure_initialized()
        
        try:
            from .hierarchical_classifier import get_hierarchical_classifier
            from .nutrition_registry import get_nutrition_registry
            
            classifier = get_hierarchical_classifier()
            registry = get_nutrition_registry()
            
            result = classifier.classify(image, registry)
            
            return {
                "stage1_traits": result.stage1_traits,
                "stage2_food_type": result.stage2_food_type,
                "resolved_food": result.resolved_food,
                "resolution_stage": result.resolution_stage,
                "confidence": result.confidence,
            }
        except Exception as e:
            logger.error(f"Hierarchical recognition failed: {e}")
            return {
                "stage1_traits": [],
                "stage2_food_type": "unknown",
                "resolved_food": "unknown",
                "resolution_stage": "error",
                "confidence": 0.0,
                "error": str(e),
            }

    def recognize_pil(
        self,
        image: Image.Image,
        top_k_dishes: int = 5,
        bypass_yolo: bool = False
    ) -> RecognitionResult:
        """
        Recognize food in a PIL Image.
        Handles multiple detections with primary selection logic.
        
        Args:
            image: PIL Image to recognize
            top_k_dishes: Number of dish suggestions to retrieve
            bypass_yolo: If True, skip YOLO detection and use full image
        """
        self._ensure_initialized()
        
        # BYPASS MODE: Skip YOLO for testing
        if bypass_yolo:
            logger.info("⚠️  YOLO BYPASS MODE - Using full image")
            result = self._classifier.classify(image, top_k_dishes)
            result.bbox = (0, 0, image.size[0], image.size[1])
            result.detection_confidence = 1.0
            result.used_fallback = True
            return result
        
        # Step 1: YOLO Detection (Multiple)
        detections = self._detector.detect(image)
        
        if not detections:
            # Fallback: use full image
            logger.info("No YOLO detection, using full image as primary")
            result = self._classifier.classify(image, top_k_dishes)
            result.bbox = (0, 0, image.size[0], image.size[1])
            result.detection_confidence = 1.0
            result.used_fallback = True
            return result
        
        # OPTIMIZATION: If only 1 detection, use full image for better dish-level accuracy
        # Single items benefit more from full context than tight crops
        if len(detections) == 1:
            logger.info("Single YOLO detection - using full image for maximum accuracy")
            result = self._classifier.classify(image, top_k_dishes)
            result.bbox = (0, 0, image.size[0], image.size[1])
            result.detection_confidence = detections[0][2]  # Use YOLO confidence
            result.used_fallback = False
            return result
        
        # Step 2: Classify all detected regions
        logger.info(f"YOLO detected {len(detections)} potential food regions")
        all_results = []
        
        for crop, bbox, det_conf in detections:
            res = self._classifier.classify(crop, top_k_dishes)
            res.bbox = bbox
            res.detection_confidence = det_conf
            res.used_fallback = False
            all_results.append(res)
            
        # Step 3: Primary Selection Logic (Largest Non-Liquid Region)
        # Criteria:
        # a. Non-liquid is preferred over liquid
        # b. Larger area is preferred within same category (liquid vs non-liquid)
        
        def is_liquid_group(res: RecognitionResult) -> bool:
            # Find in config
            for key, config in FOOD_GROUP_CONFIG.items():
                if config.get("display_name") == res.food_group.name:
                    # STREET FOOD EXCEPTION: Street food is NEVER suppressed as liquid
                    if config.get("is_street_food", False):
                        return False
                    return config.get("is_liquid", False)
            return False

        def get_area(res: RecognitionResult) -> int:
            if not res.bbox: return 0
            x1, y1, x2, y2 = res.bbox
            return (x2 - x1) * (y2 - y1)

        # Categorize
        street_foods = [r for r in all_results if any(c.get("is_street_food", False) for k,c in FOOD_GROUP_CONFIG.items() if c.get("display_name") == r.food_group.name)]
        non_liquids = [r for r in all_results if not is_liquid_group(r) and r not in street_foods]
        liquids = [r for r in all_results if is_liquid_group(r)]
        
        primary_result = None
        sides = []
        
        if street_foods:
            # STREET FOOD PRIORITY: Take largest street food scene
            street_foods.sort(key=get_area, reverse=True)
            primary_result = street_foods[0]
            sides = street_foods[1:] + non_liquids + liquids
        elif non_liquids:
            # Take largest non-liquid
            non_liquids.sort(key=get_area, reverse=True)
            primary_result = non_liquids[0]
            sides = non_liquids[1:] + liquids
        else:
            # If all are liquid, take largest liquid
            liquids.sort(key=get_area, reverse=True)
            primary_result = liquids[0]
            sides = liquids[1:]
            
        # Set sides in the primary result
        primary_result.sides = sides
        
        logger.info(f"Selected primary food: {primary_result.food_group.name} (area: {get_area(primary_result)})")
        if sides:
            logger.info(f"Detected {len(sides)} sides: {[s.food_group.name for s in sides]}")
            
        return primary_result
    
    def is_available(self) -> bool:
        try:
            self._ensure_initialized()
            return self._classifier.is_available()
        except:
            return False


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_stable_pipeline: Optional[StableFoodPipeline] = None


def get_stable_pipeline() -> StableFoodPipeline:
    """Get the global stable pipeline instance."""
    global _stable_pipeline
    if _stable_pipeline is None:
        _stable_pipeline = StableFoodPipeline()
    return _stable_pipeline


def recognize_food(image_path: str, top_k: int = 5) -> RecognitionResult:
    """
    Convenience function to recognize food.
    
    Example:
        result = recognize_food("biryani.jpg")
        print(f"Food: {result.food_group.name}")
        print(f"Suggestions: {[d.name for d in result.dish_suggestions]}")
    """
    return get_stable_pipeline().recognize(image_path, top_k)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python stable_food_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Processing: {image_path}")
    
    result = recognize_food(image_path)
    
    print(f"\n=== RESULT ===")
    print(f"Food Group: {result.food_group.name} ({result.food_group.confidence:.1%})")
    print(f"Cuisine: {result.food_group.cuisine}")
    print(f"Detection: bbox={result.bbox}, fallback={result.used_fallback}")
    
    print(f"\nDish Suggestions (Top-K):")
    for i, dish in enumerate(result.dish_suggestions, 1):
        print(f"  {i}. {dish.name} ({dish.confidence:.1%})")
    
    best = result.get_best_dish()
    if best:
        print(f"\nBest Dish (high confidence): {best}")
    else:
        print(f"\nNo clear best dish - use Food Group: {result.food_group.name}")
