"""
YOLOv11n Food Detection Service

This module provides a minimal, fast food region detector using YOLOv11n.
Its ONLY responsibility is detecting and cropping food regions - NO classification.

Architecture:
- Model: YOLOv11n (nano) for fastest inference
- Purpose: Detection and cropping ONLY
- Output: Bounding boxes and cropped PIL Images

Speed Optimizations:
1. Uses YOLOv11n (nano variant) for minimal latency
2. GPU acceleration when available
3. Minimal image size (416px default for speed)
4. Returns PIL Images directly for CLIP pipeline

IMPORTANT: This module does NOT classify food.
Classification is handled by OpenCLIP (food_recognition_openclip.py).
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DetectionResult:
    """Result from YOLO food detection."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str  # YOLO class name (may be generic like "food")
    crop: Optional[Image.Image] = None  # Cropped region


@dataclass
class DetectionOutput:
    """Full output from food detection."""
    detections: List[DetectionResult]
    num_detections: int
    image_size: Tuple[int, int]  # (width, height)


# =============================================================================
# YOLO FOOD DETECTOR
# =============================================================================

class YOLOFoodDetector:
    """
    Minimal YOLOv11n-based food region detector.
    
    This class ONLY detects food regions and returns bounding boxes/crops.
    It does NOT perform food classification - that's OpenCLIP's job.
    
    Speed Optimizations:
    1. Uses YOLOv11n (nano) - fastest YOLO variant
    2. Small inference size (416px default)
    3. GPU acceleration when available
    4. Lazy model loading (loads on first use)
    
    Usage:
        detector = YOLOFoodDetector()
        output = detector.detect(image_path)
        for detection in output.detections:
            print(f"Found food at {detection.bbox}")
            # Pass detection.crop to OpenCLIP for classification
    """
    
    # Default model paths to search
    MODEL_SEARCH_PATHS = [
        "yolo11n.pt",
        "yolov11n.pt",
        "models/yolo11n.pt",
        "models/yolov11n.pt",
    ]
    
    # COCO food-related class IDs (for filtering if using general model)
    FOOD_CLASS_IDS = {
        46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake"
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        imgsz: int = 416,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25
    ):
        """
        Initialize YOLO food detector.
        
        Args:
            model_path: Path to YOLOv11n model file
            imgsz: Inference image size (416 for speed, 640 for accuracy)
            device: Device to use ('cuda', 'cpu', or None for auto)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Lazy loading
        self.model = None
        self._model_loaded = False
        self._model_type = None  # "detection" or "segmentation"
        
        # Project root for path resolution
        self._project_root = Path(__file__).parent.parent.parent
    
    def _find_model(self) -> Optional[str]:
        """Find YOLOv11n model in common locations."""
        if self.model_path and Path(self.model_path).exists():
            return self.model_path
        
        # Search in project directories
        for path in self.MODEL_SEARCH_PATHS:
            full_path = self._project_root / path
            if full_path.exists():
                return str(full_path)
        
        # Check models directory
        models_dir = self._project_root / "models"
        if models_dir.exists():
            for file in models_dir.glob("*.pt"):
                if "yolo11n" in file.name.lower() or "yolov11n" in file.name.lower():
                    return str(file)
        
        return None
    
    def _load_model(self) -> None:
        """Load the YOLO model (called on first use)."""
        if self._model_loaded:
            return
        
        try:
            from ultralytics import YOLO
            
            # Find model
            model_path = self._find_model()
            
            if model_path is None:
                # Download yolo11n.pt automatically
                logger.info("YOLOv11n model not found, downloading...")
                model_path = "yolo11n.pt"
            
            logger.info(f"Loading YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            
            # Determine model type
            if hasattr(self.model, 'task'):
                self._model_type = self.model.task
            else:
                self._model_type = "detect"
            
            self._model_loaded = True
            logger.info(f"YOLO model loaded successfully (type: {self._model_type})")
            
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(
        self, 
        image_path: str,
        return_crops: bool = True
    ) -> DetectionOutput:
        """
        Detect food regions in an image.
        
        Args:
            image_path: Path to image file
            return_crops: If True, include cropped PIL Images in results
            
        Returns:
            DetectionOutput with list of detections
        """
        # Ensure model is loaded
        self._load_model()
        
        try:
            # Load image for cropping
            pil_image = Image.open(image_path).convert("RGB")
            image_size = pil_image.size
            
            # Run YOLO inference
            # SPEED: Use specified image size and confidence threshold
            results = self.model(
                image_path,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False  # SPEED: Suppress logs
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Get confidence
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Get class name
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        if hasattr(self.model, 'names'):
                            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        else:
                            class_name = self.FOOD_CLASS_IDS.get(cls_id, f"class_{cls_id}")
                        
                        # Create crop if requested
                        crop = None
                        if return_crops:
                            # Ensure bbox is within image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(image_size[0], x2)
                            y2 = min(image_size[1], y2)
                            crop = pil_image.crop((x1, y1, x2, y2))
                        
                        detections.append(DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=class_name,
                            crop=crop
                        ))
            
            return DetectionOutput(
                detections=detections,
                num_detections=len(detections),
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            # Return empty result on error
            return DetectionOutput(
                detections=[],
                num_detections=0,
                image_size=(0, 0)
            )
    
    def detect_from_pil(
        self, 
        image: Image.Image,
        return_crops: bool = True
    ) -> DetectionOutput:
        """
        Detect food regions from a PIL Image.
        
        Args:
            image: PIL Image
            return_crops: If True, include cropped PIL Images in results
            
        Returns:
            DetectionOutput with list of detections
        """
        # Ensure model is loaded
        self._load_model()
        
        try:
            image_size = image.size
            
            # Run YOLO inference directly on PIL image
            results = self.model(
                image,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        if hasattr(self.model, 'names'):
                            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        else:
                            class_name = self.FOOD_CLASS_IDS.get(cls_id, f"class_{cls_id}")
                        
                        crop = None
                        if return_crops:
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(image_size[0], x2)
                            y2 = min(image_size[1], y2)
                            crop = image.crop((x1, y1, x2, y2))
                        
                        detections.append(DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=class_name,
                            crop=crop
                        ))
            
            return DetectionOutput(
                detections=detections,
                num_detections=len(detections),
                image_size=image_size
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return DetectionOutput(
                detections=[],
                num_detections=0,
                image_size=image.size if image else (0, 0)
            )
    
    def get_crops(
        self, 
        image_path: str,
        max_crops: int = 4
    ) -> List[Image.Image]:
        """
        Get food region crops from an image.
        
        This is a convenience method for the CLIP pipeline.
        
        Args:
            image_path: Path to image file
            max_crops: Maximum number of crops to return (SPEED: keep low)
            
        Returns:
            List of cropped PIL Images
        """
        output = self.detect(image_path, return_crops=True)
        
        # Sort by confidence and take top-K
        detections = sorted(
            output.detections, 
            key=lambda d: d.confidence, 
            reverse=True
        )[:max_crops]
        
        return [d.crop for d in detections if d.crop is not None]
    
    def is_available(self) -> bool:
        """Check if detector is ready."""
        try:
            self._load_model()
            return self._model_loaded
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "model_loaded": self._model_loaded,
            "model_type": self._model_type,
            "device": self.device,
            "imgsz": self.imgsz,
            "confidence_threshold": self.confidence_threshold,
        }


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_detector_instance: Optional[YOLOFoodDetector] = None


def get_yolo_detector(
    model_path: Optional[str] = None,
    imgsz: int = 416
) -> YOLOFoodDetector:
    """
    Get or create the global YOLO detector instance.
    
    Args:
        model_path: Path to YOLOv11n model file
        imgsz: Inference image size
        
    Returns:
        YOLOFoodDetector singleton instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = YOLOFoodDetector(
            model_path=model_path,
            imgsz=imgsz
        )
    
    return _detector_instance


def detect_food_regions(image_path: str) -> DetectionOutput:
    """
    Convenience function to detect food regions.
    
    Args:
        image_path: Path to image file
        
    Returns:
        DetectionOutput with bounding boxes and crops
    """
    detector = get_yolo_detector()
    return detector.detect(image_path)
