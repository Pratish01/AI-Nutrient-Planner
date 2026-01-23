"""
Unified Food Recognition Pipeline

This module combines YOLOv11n detection with OpenCLIP classification
into a single, easy-to-use pipeline for food recognition.

Pipeline Flow:
1. YOLOv11n detects food regions in the image
2. Crops are extracted from detected regions
3. OpenCLIP classifies each crop hierarchically (Cuisine → Food Group)
4. Results are aggregated and returned

Speed Optimizations:
1. YOLO runs ONCE on full image for detection
2. OpenCLIP uses PRECOMPUTED text embeddings (no per-image text encoding)
3. Batch processing of multiple crops
4. GPU acceleration throughout
5. Fallback to full image if no detections

IMPORTANT:
- YOLOv11n is ONLY used for detection/cropping
- OpenCLIP is ONLY used for classification
- DO NOT mix their responsibilities
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FoodItem:
    """Recognized food item from the pipeline."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_confidence: float  # YOLO confidence
    cuisine: str
    cuisine_confidence: float
    food_group: str
    food_group_confidence: float
    # Dish name from expanded_indian_food_3000_plus.txt
    dish: Optional[str] = None
    dish_confidence: Optional[float] = None
    top_food_groups: List[Dict] = field(default_factory=list)
    top_dishes: List[Dict] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Full result from food recognition pipeline."""
    success: bool
    food_items: List[FoodItem]
    num_items: int
    processing_time_ms: float
    image_size: Tuple[int, int]
    fallback_used: bool = False  # True if YOLO detection failed
    error: Optional[str] = None


# =============================================================================
# FOOD RECOGNITION PIPELINE
# =============================================================================

class FoodRecognitionPipeline:
    """
    Unified pipeline for food recognition using YOLOv11n + OpenCLIP.
    
    This is the main entry point for food recognition in the application.
    
    Architecture:
        Input Image
            ↓
        [YOLOv11n Detection] → Bounding boxes
            ↓
        [Crop Extraction] → PIL Images
            ↓
        [OpenCLIP Classification] → Cuisine + Food Group
            ↓
        Aggregated Results
    
    Speed Optimizations:
    - YOLO nano model (fastest variant)
    - Precomputed text embeddings in OpenCLIP
    - torch.no_grad() everywhere
    - GPU acceleration
    - Small batch sizes (1-4 crops)
    
    Usage:
        pipeline = FoodRecognitionPipeline()
        result = pipeline.process_image("path/to/food.jpg")
        for item in result.food_items:
            print(f"{item.food_group} ({item.cuisine})")
    """
    
    def __init__(
        self,
        yolo_imgsz: int = 416,
        yolo_confidence: float = 0.25,
        max_crops: int = 4,
        enable_dish_classification: bool = False,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the food recognition pipeline.
        
        Args:
            yolo_imgsz: YOLO inference size (416 for speed, 640 for accuracy)
            yolo_confidence: Minimum YOLO detection confidence
            max_crops: Maximum number of crops to process (SPEED: keep low)
            enable_dish_classification: Enable dish-level prediction (slower)
            data_dir: Directory containing TXT label files
        """
        self.yolo_imgsz = yolo_imgsz
        self.yolo_confidence = yolo_confidence
        self.max_crops = max_crops
        # Always enable dish classification to get food names from TXT file
        self.enable_dish_classification = True
        self.data_dir = data_dir
        
        # Lazy loading of components
        self._detector = None
        self._classifier = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Initialize detector and classifier on first use."""
        if self._initialized:
            return
        
        logger.info("Initializing Food Recognition Pipeline...")
        start = time.time()
        
        # Import here to allow lazy loading
        from .yolo_detector import YOLOFoodDetector
        from .food_recognition_openclip import OpenCLIPClassifier
        
        # Initialize YOLO detector
        logger.info("Loading YOLOv11n detector...")
        self._detector = YOLOFoodDetector(
            imgsz=self.yolo_imgsz,
            confidence_threshold=self.yolo_confidence
        )
        
        # Initialize OpenCLIP classifier
        logger.info("Loading OpenCLIP classifier...")
        self._classifier = OpenCLIPClassifier(
            data_dir=self.data_dir,
            enable_dish_classification=self.enable_dish_classification
        )
        
        self._initialized = True
        logger.info(f"Pipeline initialized in {(time.time() - start)*1000:.0f}ms")
    
    def process_image(
        self,
        image_path: str,
        top_k: int = 3
    ) -> PipelineResult:
        """
        Process an image through the full pipeline.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions per classification level
            
        Returns:
            PipelineResult with all detected and classified food items
        """
        start_time = time.time()
        
        try:
            # Ensure components are loaded
            self._ensure_initialized()
            
            # Load image
            pil_image = Image.open(image_path).convert("RGB")
            image_size = pil_image.size
            
            # Step 1: YOLO Detection
            detection_output = self._detector.detect_from_pil(
                pil_image, 
                return_crops=True
            )
            
            food_items = []
            fallback_used = False
            
            if detection_output.num_detections > 0:
                # Process detected crops
                # SPEED: Limit to max_crops
                detections = sorted(
                    detection_output.detections,
                    key=lambda d: d.confidence,
                    reverse=True
                )[:self.max_crops]
                
                for detection in detections:
                    if detection.crop is None:
                        continue
                    
                    # Step 2: OpenCLIP Classification on crop
                    classification = self._classifier.classify_hierarchical(
                        detection.crop,
                        top_k=top_k,
                        include_dish=self.enable_dish_classification
                    )
                    
                    food_items.append(FoodItem(
                        bbox=detection.bbox,
                        detection_confidence=detection.confidence,
                        cuisine=classification.cuisine,
                        cuisine_confidence=classification.cuisine_confidence,
                        food_group=classification.food_group,
                        food_group_confidence=classification.food_group_confidence,
                        dish=classification.dish,
                        dish_confidence=classification.dish_confidence,
                        top_food_groups=classification.top_k_food_groups or [],
                        top_dishes=classification.top_k_dishes or []
                    ))
            else:
                # Fallback: Classify full image if no detections
                logger.warning("No YOLO detections, using full image fallback")
                fallback_used = True
                
                classification = self._classifier.classify_hierarchical(
                    pil_image,
                    top_k=top_k,
                    include_dish=self.enable_dish_classification
                )
                
                # Create a synthetic "full image" food item
                food_items.append(FoodItem(
                    bbox=(0, 0, image_size[0], image_size[1]),
                    detection_confidence=1.0,  # Full image
                    cuisine=classification.cuisine,
                    cuisine_confidence=classification.cuisine_confidence,
                    food_group=classification.food_group,
                    food_group_confidence=classification.food_group_confidence,
                    dish=classification.dish,
                    dish_confidence=classification.dish_confidence,
                    top_food_groups=classification.top_k_food_groups or [],
                    top_dishes=classification.top_k_dishes or []
                ))
            
            processing_time = (time.time() - start_time) * 1000
            
            return PipelineResult(
                success=True,
                food_items=food_items,
                num_items=len(food_items),
                processing_time_ms=processing_time,
                image_size=image_size,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            processing_time = (time.time() - start_time) * 1000
            return PipelineResult(
                success=False,
                food_items=[],
                num_items=0,
                processing_time_ms=processing_time,
                image_size=(0, 0),
                error=str(e)
            )
    
    def process_pil_image(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> PipelineResult:
        """
        Process a PIL Image through the pipeline.
        
        Args:
            image: PIL Image to process
            top_k: Number of top predictions per level
            
        Returns:
            PipelineResult with all detected and classified food items
        """
        start_time = time.time()
        
        try:
            self._ensure_initialized()
            
            image_size = image.size
            
            # Step 1: YOLO Detection
            detection_output = self._detector.detect_from_pil(
                image, 
                return_crops=True
            )
            
            food_items = []
            fallback_used = False
            
            if detection_output.num_detections > 0:
                detections = sorted(
                    detection_output.detections,
                    key=lambda d: d.confidence,
                    reverse=True
                )[:self.max_crops]
                
                for detection in detections:
                    if detection.crop is None:
                        continue
                    
                    classification = self._classifier.classify_hierarchical(
                        detection.crop,
                        top_k=top_k,
                        include_dish=self.enable_dish_classification
                    )
                    
                    food_items.append(FoodItem(
                        bbox=detection.bbox,
                        detection_confidence=detection.confidence,
                        cuisine=classification.cuisine,
                        cuisine_confidence=classification.cuisine_confidence,
                        food_group=classification.food_group,
                        food_group_confidence=classification.food_group_confidence,
                        dish=classification.dish,
                        dish_confidence=classification.dish_confidence,
                        top_food_groups=classification.top_k_food_groups or [],
                        top_dishes=classification.top_k_dishes or []
                    ))
            else:
                logger.warning("No YOLO detections, using full image fallback")
                fallback_used = True
                
                classification = self._classifier.classify_hierarchical(
                    image,
                    top_k=top_k,
                    include_dish=self.enable_dish_classification
                )
                
                food_items.append(FoodItem(
                    bbox=(0, 0, image_size[0], image_size[1]),
                    detection_confidence=1.0,
                    cuisine=classification.cuisine,
                    cuisine_confidence=classification.cuisine_confidence,
                    food_group=classification.food_group,
                    food_group_confidence=classification.food_group_confidence,
                    dish=classification.dish,
                    dish_confidence=classification.dish_confidence,
                    top_food_groups=classification.top_k_food_groups or [],
                    top_dishes=classification.top_k_dishes or []
                ))
            
            processing_time = (time.time() - start_time) * 1000
            
            return PipelineResult(
                success=True,
                food_items=food_items,
                num_items=len(food_items),
                processing_time_ms=processing_time,
                image_size=image_size,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            processing_time = (time.time() - start_time) * 1000
            return PipelineResult(
                success=False,
                food_items=[],
                num_items=0,
                processing_time_ms=processing_time,
                image_size=image.size if image else (0, 0),
                error=str(e)
            )
    
    def classify_only(
        self,
        image: Image.Image,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Classify an image without YOLO detection (for pre-cropped images).
        
        Args:
            image: PIL Image (should already be cropped to food region)
            top_k: Number of top predictions
            
        Returns:
            Classification result dict
        """
        self._ensure_initialized()
        
        result = self._classifier.classify_hierarchical(
            image,
            top_k=top_k,
            include_dish=self.enable_dish_classification
        )
        
        return {
            "cuisine": result.cuisine,
            "cuisine_confidence": result.cuisine_confidence,
            "food_group": result.food_group,
            "food_group_confidence": result.food_group_confidence,
            "dish": result.dish,
            "dish_confidence": result.dish_confidence,
            "top_food_groups": result.top_k_food_groups,
            "top_dishes": result.top_k_dishes
        }
    
    def check_and_reload_labels(self) -> bool:
        """
        Check if TXT label files have changed and reload if needed.
        
        Returns:
            True if labels were reloaded
        """
        if self._classifier:
            return self._classifier.check_and_reload_if_needed()
        return False
    
    def is_available(self) -> bool:
        """Check if pipeline is ready."""
        try:
            self._ensure_initialized()
            return (
                self._detector is not None and 
                self._classifier is not None and
                self._classifier.is_available()
            )
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "initialized": self._initialized,
            "yolo_imgsz": self.yolo_imgsz,
            "max_crops": self.max_crops,
            "dish_classification_enabled": self.enable_dish_classification,
        }
        
        if self._detector:
            stats["detector"] = self._detector.get_stats()
        if self._classifier:
            stats["classifier"] = self._classifier.get_stats()
        
        return stats


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_pipeline_instance: Optional[FoodRecognitionPipeline] = None


def get_food_pipeline(
    yolo_imgsz: int = 416,
    max_crops: int = 4,
    enable_dish_classification: bool = True  # Now defaults to True
) -> FoodRecognitionPipeline:
    """
    Get or create the global food recognition pipeline instance.
    
    Args:
        yolo_imgsz: YOLO inference size
        max_crops: Maximum crops to process
        enable_dish_classification: Enable dish-level prediction
        
    Returns:
        FoodRecognitionPipeline singleton instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = FoodRecognitionPipeline(
            yolo_imgsz=yolo_imgsz,
            max_crops=max_crops,
            enable_dish_classification=enable_dish_classification
        )
    
    return _pipeline_instance


def recognize_food(image_path: str, top_k: int = 3) -> PipelineResult:
    """
    Convenience function to recognize food in an image.
    
    Args:
        image_path: Path to image file
        top_k: Number of predictions per level
        
    Returns:
        PipelineResult with detected food items
    """
    pipeline = get_food_pipeline()
    return pipeline.process_image(image_path, top_k=top_k)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python food_pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"Processing: {image_path}")
    
    # Create pipeline
    pipeline = FoodRecognitionPipeline()
    
    # Process image
    result = pipeline.process_image(image_path)
    
    print(f"\nResults (processed in {result.processing_time_ms:.0f}ms):")
    print(f"Success: {result.success}")
    print(f"Items detected: {result.num_items}")
    print(f"Fallback used: {result.fallback_used}")
    
    for i, item in enumerate(result.food_items):
        print(f"\nItem {i+1}:")
        print(f"  Dish: {item.dish} ({item.dish_confidence:.2%})" if item.dish else "  Dish: Unknown")
        print(f"  Cuisine: {item.cuisine} ({item.cuisine_confidence:.2%})")
        print(f"  Food Group: {item.food_group} ({item.food_group_confidence:.2%})")
        print(f"  Bbox: {item.bbox}")
        if item.top_dishes:
            print(f"  Top dishes: {item.top_dishes[:3]}")
