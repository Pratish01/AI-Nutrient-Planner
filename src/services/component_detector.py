"""
CLIP-Based Visual Component Detector

Uses OpenCLIP to perform binary detection of visual components in food images.
Each component is scored independently against its CLIP prompt.
"""

from typing import Dict, Set, Tuple, Optional
from PIL import Image
import torch
import logging

from .component_rules import (
    VisualComponent,
    FoodStructure,
    COMPONENT_PROMPTS,
    ComponentDetectionResult,
)

logger = logging.getLogger(__name__)


class CLIPComponentDetector:
    """
    Detects visual components in food images using CLIP embeddings.
    
    Each component is evaluated independently using a binary threshold.
    This avoids softmax competition between unrelated components.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        threshold: float = 0.25,  # Binary detection threshold
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = threshold
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = None
        
        # Pre-computed text embeddings for components
        self._component_embeddings: Optional[torch.Tensor] = None
        self._component_order: list = []
        
    def _ensure_initialized(self):
        """Lazy initialization of CLIP model."""
        if self.model is not None:
            return
            
        import open_clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[ComponentDetector] Loading CLIP on {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Pre-compute component embeddings
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        """Pre-compute text embeddings for all components."""
        prompts = []
        self._component_order = []
        
        for component, prompt in COMPONENT_PROMPTS.items():
            self._component_order.append(component)
            prompts.append(prompt)
        
        tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        self._component_embeddings = embeddings
        logger.info(f"[ComponentDetector] Pre-computed {len(prompts)} component embeddings")
    
    def detect(self, image: Image.Image) -> ComponentDetectionResult:
        """
        Detect visual components in an image.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            ComponentDetectionResult with detected components and structure
        """
        self._ensure_initialized()
        
        # Encode image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all components
        similarities = (image_embedding @ self._component_embeddings.T).squeeze(0)
        
        # Convert to probabilities using sigmoid (independent, not softmax)
        # This allows multiple components to be detected simultaneously
        probs = torch.sigmoid(similarities * 10)  # Scale for sharper decisions
        
        # Binary detection
        detected: Set[VisualComponent] = set()
        confidences: Dict[VisualComponent, float] = {}
        
        for i, component in enumerate(self._component_order):
            conf = float(probs[i].cpu())
            confidences[component] = conf
            
            if conf >= self.threshold:
                detected.add(component)
        
        # Log detected components
        if detected:
            logger.info(f"[ComponentDetector] Detected: {[c.value for c in detected]}")
        else:
            logger.info("[ComponentDetector] No components detected above threshold")
        
        # Determine structure
        structure = self._analyze_structure(detected)
        
        return ComponentDetectionResult(
            detected=detected,
            confidences=confidences,
            structure=structure,
        )
    
    def _analyze_structure(self, components: Set[VisualComponent]) -> FoodStructure:
        """Analyze food structure from detected components."""
        has_liquid = any([
            VisualComponent.SPICED_LIQUID in components,
            VisualComponent.THICK_CREAMY_GRAVY in components,
            VisualComponent.WATERY_GRAVY in components,
            VisualComponent.SAMBAR in components,
        ])
        
        has_dry = any([
            VisualComponent.HOLLOW_SHELL in components,
            VisualComponent.SOLID_FRIED_PATTY in components,
            VisualComponent.PUFFED_RICE in components,
            VisualComponent.FLAT_BREAD in components,
            VisualComponent.LONG_GRAIN_RICE in components,
        ])
        
        is_snack = VisualComponent.SMALL_SNACK_SCALE in components
        
        if has_liquid and has_dry:
            return FoodStructure.DRY_PLUS_LIQUID
        elif has_liquid:
            return FoodStructure.LIQUID_ONLY
        elif has_dry:
            return FoodStructure.DRY_ONLY
        elif is_snack:
            return FoodStructure.MIXED_SCENE
        else:
            return FoodStructure.UNKNOWN
    
    def get_top_components(
        self,
        image: Image.Image,
        top_k: int = 10
    ) -> list:
        """
        Get top-k components by confidence for debugging.
        
        Returns:
            List of (component, confidence) tuples
        """
        self._ensure_initialized()
        
        result = self.detect(image)
        
        sorted_components = sorted(
            result.confidences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [(c.value, conf) for c, conf in sorted_components[:top_k]]


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_detector_instance: Optional[CLIPComponentDetector] = None

def get_component_detector() -> CLIPComponentDetector:
    """Get or create the component detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CLIPComponentDetector()
    return _detector_instance
