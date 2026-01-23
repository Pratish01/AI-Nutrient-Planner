"""
YOLO Food Recognition Service

Uses YOLOv11n-seg model for food detection and segmentation in images.
Integrates with the nutrition pipeline to identify foods from uploaded images.
Supports both detection and segmentation modes.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path


class YOLOFoodRecognizer:
    """
    YOLO-based food recognition service.
    
    Uses the yolo11n-seg.pt model for detection and segmentation.
    Falls back to best.pt if segmentation model not available.
    
    Optimizations:
    - FP16 half-precision on GPU for ~2x speedup
    - Configurable image size (smaller = faster)
    - Optional mask computation
    """
    
    # Default inference settings for optimization
    DEFAULT_IMGSZ = 416  # Smaller than 640 default for faster inference
    USE_HALF_PRECISION = True  # FP16 on GPU
    
    def __init__(self, model_path: Optional[str] = None, imgsz: int = None):
        """
        Initialize YOLO food recognizer.
        
        Args:
            model_path: Path to YOLO model file
            imgsz: Inference image size (default 416 for speed, use 640 for accuracy)
        """
        self.model = None
        self.model_path = model_path or self._find_model()
        self._initialized = False
        self.imgsz = imgsz or self.DEFAULT_IMGSZ
        self._use_half = False  # Set during model loading based on GPU availability
        
        # Don't load model immediately, do it lazily
    
    def _find_model(self) -> str:
        """Find the YOLO model in common locations."""
        # Prefer segmentation model for portion estimation
        locations = [
            # Segmentation model (preferred)
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo11n-seg.pt"),
            # Detection model (fallback)
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "best.pt"),
            os.path.join(os.path.dirname(__file__), "..", "models", "yolo11n-seg.pt"),
            os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"),
            "models/yolo11n-seg.pt",
            "models/best.pt",
        ]
        
        for loc in locations:
            if os.path.exists(loc):
                return os.path.abspath(loc)
        
        return os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo11n-seg.pt")
    
    def _ensure_model_loaded(self):
        """Load the YOLO model only when needed."""
        if self._initialized:
            return

        try:
            import torch
            from ultralytics import YOLO
            
            if os.path.exists(self.model_path):
                print(f"[YOLO] Loading model lazily from: {self.model_path}...")
                self.model = YOLO(self.model_path)
                self._initialized = True
                
                # Enable FP16 on CUDA for ~2x speedup
                self._use_half = torch.cuda.is_available() and self.USE_HALF_PRECISION
                if self._use_half:
                    print(f"[YOLO] FP16 half-precision enabled (GPU detected)")
                
                print(f"[YOLO] Model loaded successfully (imgsz={self.imgsz})")
                
                # Print model info
                if hasattr(self.model, 'names'):
                    print(f"[YOLO] Task: {self.model.task}")
                    print(f"[YOLO] Classes: {len(self.model.names)} classes available")
            else:
                print(f"[YOLO] Model file not found: {self.model_path}")
                
        except ImportError:
            print("[YOLO] ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if model is ready (trigger load if needed)."""
        self._ensure_model_loaded()
        return self._initialized and self.model is not None
    
    @property
    def supports_segmentation(self) -> bool:
        """Check if model supports segmentation."""
        if not self.is_available:
            return False
        return self.model.task in ["segment", "seg"]
    
    def predict(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Detect and optionally segment food items in an image.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dict with detected/classified food name, confidence, masks, and all predictions
        """
        if not self.is_available:
            return {
                "success": False,
                "error": "YOLO model not available",
                "food_name": None,
                "confidence": 0.0,
                "detections": []
            }
        
        try:
            # Run optimized inference
            print(f"[YOLO] Running prediction on: {image_path}")
            print(f"[YOLO] Model task: {self.model.task}, imgsz={self.imgsz}, half={self._use_half}")
            results = self.model(
                image_path,
                verbose=False,
                imgsz=self.imgsz,
                half=self._use_half,  # FP16 for GPU speedup
                retina_masks=False,   # Skip high-res masks for speed
            )
            
            # Handle CLASSIFICATION models (uses result.probs)
            if self.model.task == "classify":
                return self._handle_classification(results, confidence_threshold)
            
            # Handle DETECTION and SEGMENTATION models (uses result.boxes and result.masks)
            return self._handle_detection_segmentation(results, confidence_threshold)
                
        except Exception as e:
            print(f"[YOLO] Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "food_name": None,
                "confidence": 0.0,
                "detections": []
            }
    
    def _handle_classification(self, results, confidence_threshold: float) -> Dict[str, Any]:
        """Handle classification model output."""
        for result in results:
            probs = result.probs
            if probs is not None:
                top1_idx = probs.top1
                top1_conf = float(probs.top1conf)
                class_name = self.model.names[top1_idx]
                
                # Get top 5 for debugging
                top5_indices = probs.top5
                top5_confs = probs.top5conf.tolist()
                all_predictions = [f"{self.model.names[idx]}: {conf:.1%}" 
                                  for idx, conf in zip(top5_indices, top5_confs)]
                print(f"[YOLO] Top 5 classifications: {all_predictions}")
                
                if top1_conf >= confidence_threshold:
                    return {
                        "success": True,
                        "food_name": class_name,
                        "confidence": top1_conf,
                        "detections": [{
                            "class_name": self.model.names[idx], 
                            "confidence": conf
                        } for idx, conf in zip(top5_indices, top5_confs)],
                        "total_detections": 1,
                        "model_type": "classification"
                    }
                else:
                    return {
                        "success": True,
                        "food_name": None,
                        "confidence": top1_conf,
                        "detections": [],
                        "message": f"Best prediction ({class_name}) below confidence threshold"
                    }
        
        return {
            "success": True,
            "food_name": None,
            "confidence": 0.0,
            "detections": [],
            "message": "No classification results"
        }
    
    def _handle_detection_segmentation(self, results, confidence_threshold: float) -> Dict[str, Any]:
        """Handle detection/segmentation model output."""
        detections = []
        best_detection = None
        best_confidence = 0.0
        
        for result in results:
            boxes = result.boxes
            masks = result.masks  # Segmentation masks (if available)
            
            if boxes is None:
                print(f"[YOLO] No boxes in result")
                continue
            
            print(f"[YOLO] Found {len(boxes)} detections")
            has_masks = masks is not None and len(masks) > 0
            print(f"[YOLO] Segmentation masks available: {has_masks}")
            
            for i, box in enumerate(boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls] if cls in self.model.names else f"class_{cls}"
                
                detection = {
                    "class_name": class_name,
                    "confidence": conf,
                    "box": box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None,
                    "mask_area": None
                }
                
                # Extract mask area if segmentation is available
                if has_masks and i < len(masks):
                    try:
                        mask_data = masks[i].data.cpu().numpy()
                        mask_area = int(mask_data.sum())  # Count non-zero pixels
                        detection["mask_area"] = mask_area
                        print(f"[YOLO] Detection {i}: {class_name} ({conf:.1%}), mask_area={mask_area}")
                    except Exception as e:
                        print(f"[YOLO] Error extracting mask: {e}")
                
                if conf >= confidence_threshold:
                    detections.append(detection)
                    
                    if conf > best_confidence:
                        best_confidence = conf
                        best_detection = detection
        
        if best_detection:
            return {
                "success": True,
                "food_name": best_detection["class_name"],
                "confidence": best_detection["confidence"],
                "detections": detections,
                "total_detections": len(detections),
                "model_type": "segmentation" if self.supports_segmentation else "detection",
                "has_masks": any(d.get("mask_area") for d in detections)
            }
        else:
            return {
                "success": True,
                "food_name": None,
                "confidence": 0.0,
                "detections": [],
                "message": "No food detected above confidence threshold"
            }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        if self.is_available and hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return []


# Global singleton
_yolo_recognizer: Optional[YOLOFoodRecognizer] = None


def get_yolo_recognizer() -> YOLOFoodRecognizer:
    """Get or create the global YOLO recognizer instance."""
    global _yolo_recognizer
    if _yolo_recognizer is None:
        _yolo_recognizer = YOLOFoodRecognizer()
    return _yolo_recognizer


def recognize_food(image_path: str) -> Dict[str, Any]:
    """
    Convenience function to recognize food in an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dict with food_name, confidence, and detection details
    """
    recognizer = get_yolo_recognizer()
    return recognizer.predict(image_path)

