"""
Unified Food Recognition Pipeline

Combines YOLO segmentation, CLIP classification, and portion estimation
into a single pipeline for complete food analysis.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

# Local imports
from .yolo_service import get_yolo_recognizer
from .clip_service import get_clip_classifier
from .food_estimation_service import get_estimation_service


class FoodRecognitionPipeline:
    """
    Unified pipeline for food recognition:
    1. YOLO11n-seg for detection and segmentation
    2. CLIP for open-vocabulary classification
    3. Portion estimation from mask areas
    4. Nutrition lookup and scaling
    """
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.yolo = get_yolo_recognizer()
        self.clip = get_clip_classifier()
        self.estimator = get_estimation_service()
        
        print(f"[Pipeline] YOLO available: {self.yolo.is_available}")
        print(f"[Pipeline] CLIP available: {self.clip.is_available}")
    
    # COCO classes that are containers, not food
    CONTAINER_CLASSES = {
        'bowl', 'cup', 'plate', 'dining table', 'bottle', 
        'wine glass', 'fork', 'knife', 'spoon', 'person'
    }
    
    def process_image(
        self,
        image_path: str,
        use_clip: bool = True,
        estimate_portions: bool = True,
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Process a food image through the full pipeline.
        
        Args:
            image_path: Path to the image file
            use_clip: Whether to use CLIP for classification
            estimate_portions: Whether to estimate portion sizes
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dict with detected foods, classifications, and nutrition
        """
        results = {
            "success": False,
            "image_path": image_path,
            "detections": [],
            "primary_food": None,
            "total_nutrition": {},
            "pipeline_info": {
                "yolo_available": self.yolo.is_available,
                "clip_available": self.clip.is_available,
                "use_clip": use_clip,
                "estimate_portions": estimate_portions
            }
        }
        
        if not os.path.exists(image_path):
            results["error"] = "Image file not found"
            return results
        
        try:
            # Load image for dimensions
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Step 1: Run YOLO detection/segmentation
            yolo_result = self._run_yolo(image_path, confidence_threshold)
            
            # Check if YOLO found actual food or just containers
            yolo_found_food = False
            yolo_found_container = False
            container_detection = None
            
            if yolo_result.get("success"):
                for detection in yolo_result.get("detections", []):
                    class_name = detection.get("class_name", "").lower()
                    if class_name in self.CONTAINER_CLASSES:
                        yolo_found_container = True
                        container_detection = detection
                        print(f"[Pipeline] YOLO detected container: {class_name}")
                    else:
                        yolo_found_food = True
            
            # If YOLO only found containers (bowl, plate, etc.), use CLIP on full image
            if yolo_found_container and not yolo_found_food:
                print(f"[Pipeline] YOLO found container but not food - using CLIP on full image")
                if use_clip and self.clip.is_available:
                    clip_result = self.clip.classify_image(image, top_k=5)
                    if clip_result.get("success") and clip_result.get("predictions"):
                        top_pred = clip_result["predictions"][0]
                        print(f"[Pipeline] CLIP identified: {top_pred['food_name']} ({top_pred['confidence']:.1%})")
                        
                        food_item = {
                            "food_name": top_pred["food_name"],
                            "confidence": top_pred["confidence"],
                            "source": "clip_full_image",
                            "nutrition": top_pred.get("nutrition"),
                            "yolo_class": container_detection.get("class_name") if container_detection else None,
                            "bbox": container_detection.get("box") if container_detection else None,
                            "mask_area": container_detection.get("mask_area") if container_detection else None,
                            "clip_predictions": clip_result["predictions"][:3]
                        }
                        
                        if estimate_portions:
                            mask_area = container_detection.get("mask_area") if container_detection else int(img_width * img_height * 0.3)
                            food_item["portion_estimate"] = self.estimator.estimate_portion_from_mask(
                                mask_area or int(img_width * img_height * 0.3),
                                food_item["food_name"],
                                img_width,
                                img_height,
                                self.estimator.get_depth_estimate(food_item["food_name"])
                            )
                            
                            if food_item.get("nutrition") and food_item.get("portion_estimate", {}).get("estimated_grams"):
                                food_item["scaled_nutrition"] = self.estimator.scale_nutrition(
                                    food_item["nutrition"],
                                    food_item["portion_estimate"]["estimated_grams"]
                                )
                        
                        results["success"] = True
                        results["detections"] = [food_item]
                        results["primary_food"] = food_item
                        results["total_nutrition"] = self._aggregate_nutrition([food_item])
                        results["detection_count"] = 1
                        results["pipeline_info"]["method"] = "container_clip"
                        return results
            
            # YOLO didn't find food - fallback to CLIP-only
            if not yolo_result.get("success") or not yolo_found_food:
                if use_clip and self.clip.is_available:
                    return self._clip_only_pipeline(image_path, image)
                else:
                    results["error"] = yolo_result.get("error", "No food detected")
                    return results
            
            # Step 2: Process each detection (only non-container classes)
            detections = []
            for detection in yolo_result.get("detections", []):
                class_name = detection.get("class_name", "").lower()
                if class_name in self.CONTAINER_CLASSES:
                    continue  # Skip containers
                    
                food_item = self._process_detection(
                    detection,
                    image,
                    img_width,
                    img_height,
                    use_clip,
                    estimate_portions
                )
                if food_item:
                    detections.append(food_item)
            
            # If no food detections, try CLIP on full image
            if not detections and use_clip and self.clip.is_available:
                clip_result = self.clip.classify_image(image, top_k=3)
                if clip_result.get("success") and clip_result.get("predictions"):
                    top_pred = clip_result["predictions"][0]
                    food_item = {
                        "food_name": top_pred["food_name"],
                        "confidence": top_pred["confidence"],
                        "source": "clip_full_image",
                        "nutrition": top_pred.get("nutrition"),
                        "portion_estimate": None
                    }
                    
                    if estimate_portions:
                        food_item["portion_estimate"] = self.estimator.estimate_portion_from_mask(
                            int(img_width * img_height * 0.3),
                            food_item["food_name"],
                            img_width,
                            img_height,
                            self.estimator.get_depth_estimate(food_item["food_name"])
                        )
                    
                    detections.append(food_item)
            
            # Sort by confidence
            detections.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            # Calculate total nutrition
            total_nutrition = self._aggregate_nutrition(detections)
            
            results["success"] = True
            results["detections"] = detections
            results["primary_food"] = detections[0] if detections else None
            results["total_nutrition"] = total_nutrition
            results["detection_count"] = len(detections)
            
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
            return results
    
    def _run_yolo(self, image_path: str, confidence_threshold: float) -> Dict[str, Any]:
        """Run YOLO detection/segmentation."""
        if not self.yolo.is_available:
            return {"success": False, "error": "YOLO model not available"}
        
        try:
            result = self.yolo.predict(image_path, confidence_threshold)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_detection(
        self,
        detection: Dict[str, Any],
        image: Image.Image,
        img_width: int,
        img_height: int,
        use_clip: bool,
        estimate_portions: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single YOLO detection.
        YOLO is used ONLY for bounding boxes - CLIP does all food identification.
        """
        try:
            food_item = {
                "yolo_class": detection.get("class_name"),  # Keep for reference only
                "yolo_confidence": detection.get("confidence"),
                "bbox": detection.get("box"),
                "mask_area": detection.get("mask_area"),
            }
            
            bbox = detection.get("box")
            
            # Always use CLIP for food identification
            if use_clip and self.clip.is_available and bbox:
                print(f"[Pipeline] Using CLIP to identify food in detected region")
                
                # Add 15% padding to the crop for better CLIP context
                x1, y1, x2, y2 = bbox
                bw, bh = x2 - x1, y2 - y1
                pad_w, pad_h = bw * 0.15, bh * 0.15
                
                padded_bbox = (
                    max(0, int(x1 - pad_w)),
                    max(0, int(y1 - pad_h)),
                    min(img_width, int(x2 + pad_w)),
                    min(img_height, int(y2 + pad_h))
                )
                
                clip_result = self.clip.classify_crop(image, padded_bbox, top_k=3)
                
                if clip_result.get("success") and clip_result.get("predictions"):
                    top_pred = clip_result["predictions"][0]
                    food_item["food_name"] = top_pred["food_name"]
                    food_item["confidence"] = top_pred["confidence"]
                    food_item["nutrition"] = top_pred.get("nutrition")
                    food_item["source"] = "clip"
                    food_item["clip_predictions"] = clip_result["predictions"][:3]
                    print(f"[Pipeline] CLIP identified: {top_pred['food_name']} ({top_pred['confidence']:.1%})")
                else:
                    # CLIP failed - use generic name
                    print(f"[Pipeline] CLIP failed: {clip_result.get('error', 'No predictions')}")
                    food_item["food_name"] = "Unknown Food"
                    food_item["confidence"] = 0.5
                    food_item["source"] = "detection_only"
            else:
                # CLIP not available - use generic name
                if not use_clip:
                    print(f"[Pipeline] CLIP disabled")
                elif not self.clip.is_available:
                    print(f"[Pipeline] CLIP not available")
                food_item["food_name"] = "Detected Food"
                food_item["confidence"] = detection.get("confidence", 0.5)
                food_item["source"] = "detection_only"
            
            # Estimate portion size
            if estimate_portions:
                mask_area = detection.get("mask_area")
                if mask_area:
                    food_item["portion_estimate"] = self.estimator.estimate_portion_from_mask(
                        mask_area,
                        food_item["food_name"],
                        img_width,
                        img_height,
                        self.estimator.get_depth_estimate(food_item["food_name"])
                    )
                elif detection.get("box"):
                    food_item["portion_estimate"] = self.estimator.estimate_portion_from_bbox(
                        tuple(detection["box"]),
                        food_item["food_name"],
                        img_width,
                        img_height
                    )
                
                # Scale nutrition based on portion estimate
                if food_item.get("nutrition") and food_item.get("portion_estimate", {}).get("estimated_grams"):
                    food_item["scaled_nutrition"] = self.estimator.scale_nutrition(
                        food_item["nutrition"],
                        food_item["portion_estimate"]["estimated_grams"]
                    )
            
            # Get nutrition if not already present
            if not food_item.get("nutrition"):
                food_item["nutrition"] = self.estimator.get_nutrition(food_item["food_name"])
            
            return food_item
            
        except Exception as e:
            print(f"[Pipeline] Error processing detection: {e}")
            return None
    
    def _clip_only_pipeline(self, image_path: str, image: Image.Image) -> Dict[str, Any]:
        """Run CLIP-only classification when YOLO is not available."""
        result = {
            "success": False,
            "image_path": image_path,
            "detections": [],
            "primary_food": None,
            "total_nutrition": {},
            "pipeline_info": {"source": "clip_only"}
        }
        
        try:
            clip_result = self.clip.classify_food(image_path, top_k=5)
            
            if not clip_result.get("success"):
                result["error"] = clip_result.get("error", "CLIP classification failed")
                return result
            
            img_width, img_height = image.size
            
            detections = []
            for pred in clip_result.get("predictions", []):
                food_item = {
                    "food_name": pred["food_name"],
                    "confidence": pred["confidence"],
                    "nutrition": pred.get("nutrition"),
                    "source": "clip_only"
                }
                
                # Estimate portion (assume single food covers ~40% of image)
                portion_estimate = self.estimator.estimate_portion_from_mask(
                    int(img_width * img_height * 0.4),
                    food_item["food_name"],
                    img_width,
                    img_height,
                    self.estimator.get_depth_estimate(food_item["food_name"])
                )
                food_item["portion_estimate"] = portion_estimate
                
                if food_item.get("nutrition") and portion_estimate.get("estimated_grams"):
                    food_item["scaled_nutrition"] = self.estimator.scale_nutrition(
                        food_item["nutrition"],
                        portion_estimate["estimated_grams"]
                    )
                
                detections.append(food_item)
            
            result["success"] = True
            result["detections"] = detections
            result["primary_food"] = detections[0] if detections else None
            result["total_nutrition"] = self._aggregate_nutrition(detections[:1])  # Top 1 only
            result["detection_count"] = len(detections)
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def _aggregate_nutrition(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate nutrition from multiple detections."""
        total = {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "sugar_g": 0,
            "fiber_g": 0,
            "sodium_mg": 0
        }
        
        for det in detections:
            # Use scaled nutrition if available, else use base nutrition
            nutrition = det.get("scaled_nutrition") or det.get("nutrition")
            if nutrition:
                for key in total:
                    if key in nutrition:
                        total[key] += nutrition[key]
        
        # Round the totals
        for key in total:
            total[key] = round(total[key], 1)
        
        return total


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_pipeline: Optional[FoodRecognitionPipeline] = None


def get_food_pipeline() -> FoodRecognitionPipeline:
    """Get or create the global food recognition pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FoodRecognitionPipeline()
    return _pipeline


def recognize_food(image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function to recognize food in an image.
    
    Args:
        image_path: Path to the image file
        **kwargs: Additional options passed to process_image
        
    Returns:
        Complete food analysis results
    """
    pipeline = get_food_pipeline()
    return pipeline.process_image(image_path, **kwargs)
