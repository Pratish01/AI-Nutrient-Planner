"""
Hierarchical Food Recognition Pipeline

Full pipeline orchestrating:
1. YOLO detection for food region bounding boxes
2. CLIP Stage 1: Cuisine classification on crops
3. CLIP Stage 2: Food_Group classification filtered by cuisine
4. Nutrition lookup from CSV using (Cuisine, Food_Group)

IMPORTANT: CLIP runs ONLY on YOLO crops, NEVER on full images.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image


class HierarchicalFoodPipeline:
    """
    Main pipeline for hierarchical food recognition and nutrition estimation.
    
    Architecture:
    1. YOLO detects food regions (bounding boxes)
    2. For each crop:
       - CLIP Stage 1: Classify Cuisine (Indian/Continental)
       - CLIP Stage 2: Classify Food_Group filtered by Cuisine
       - Lookup average nutrition from CSV using (Cuisine, Food_Group)
    3. Return list of detected foods with nutrition estimates
    
    Features:
    - Precomputed CLIP embeddings for speed
    - Top-K predictions for robustness
    - Falls back to "Other" food_group if classification uncertain
    - Density values available for volume-to-weight conversion
    """
    
    def __init__(self):
        """Initialize the pipeline (lazy loading of models)."""
        self._yolo = None
        self._clip_classifier = None
        self._nutrition_registry = None
        
    def _get_yolo(self):
        """Lazy load YOLO recognizer."""
        if self._yolo is None:
            from src.services.yolo_service import get_yolo_recognizer
            self._yolo = get_yolo_recognizer()
        return self._yolo
    
    def _get_clip_classifier(self):
        """Lazy load CLIP classifier."""
        if self._clip_classifier is None:
            from src.services.hierarchical_clip_classifier import get_hierarchical_clip_classifier
            self._clip_classifier = get_hierarchical_clip_classifier()
        return self._clip_classifier
    
    def _get_nutrition_registry(self):
        """Lazy load nutrition registry."""
        if self._nutrition_registry is None:
            from src.services.nutrition_registry import get_nutrition_registry
            self._nutrition_registry = get_nutrition_registry()
        return self._nutrition_registry
    
    def process_image(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Process a food image through the full pipeline.
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum YOLO detection confidence
            
        Returns:
            Dict with:
            - success: bool
            - detected_foods: list of food items with nutrition
            - summary: aggregated nutrition totals
        """
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image not found: {image_path}"}
        
        try:
            # Step 1: YOLO detection for bounding boxes
            print(f"[Pipeline] Step 1: YOLO detection on {image_path}")
            yolo = self._get_yolo()
            
            if not yolo.is_available:
                return {"success": False, "error": "YOLO model not available"}
            
            yolo_result = yolo.predict(image_path, confidence_threshold=confidence_threshold)
            
            # Get bounding boxes from YOLO result
            bboxes = self._extract_bboxes(yolo_result)
            
            if not bboxes:
                print("[Pipeline] No food regions detected by YOLO, using full image")
                # Fallback: process the entire image
                bboxes = [None]  # None means full image
            else:
                print(f"[Pipeline] YOLO detected {len(bboxes)} food regions")
            
            # Step 2-4: Process each crop
            detected_foods = []
            original_image = Image.open(image_path).convert("RGB")
            
            for i, bbox in enumerate(bboxes):
                print(f"[Pipeline] Processing region {i+1}/{len(bboxes)}")
                
                # Crop the region (or use full image if no bbox)
                if bbox is not None:
                    crop = original_image.crop(bbox)
                else:
                    crop = original_image
                
                # Step 2-3: CLIP hierarchical classification
                clip_classifier = self._get_clip_classifier()
                
                if not clip_classifier.is_available:
                    print("[Pipeline] CLIP not available, skipping classification")
                    continue
                
                # Run two-stage classification
                clip_result = clip_classifier.classify_hierarchical(crop)
                
                if not clip_result.get("success"):
                    print(f"[Pipeline] CLIP classification failed: {clip_result.get('error')}")
                    continue
                
                cuisine = clip_result.get("cuisine")
                food_group = clip_result.get("food_group")
                
                print(f"[Pipeline] Detected: {cuisine} / {food_group}")
                
                # Step 4: Nutrition lookup
                nutrition = self._lookup_nutrition(cuisine, food_group)
                
                # Build detected food item
                food_item = {
                    "bbox": bbox,
                    "cuisine": cuisine,
                    "cuisine_confidence": clip_result.get("cuisine_confidence"),
                    "food_group": food_group,
                    "food_group_confidence": clip_result.get("food_group_confidence"),
                    "nutrition_per_100g": nutrition,
                    "cuisine_predictions": clip_result.get("cuisine_predictions"),
                    "food_group_predictions": clip_result.get("food_group_predictions")
                }
                
                detected_foods.append(food_item)
            
            # Calculate summary
            summary = self._calculate_summary(detected_foods)
            
            return {
                "success": True,
                "image_path": image_path,
                "detected_foods": detected_foods,
                "total_detections": len(detected_foods),
                "summary": summary
            }
            
        except Exception as e:
            print(f"[Pipeline] Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _extract_bboxes(self, yolo_result: Dict[str, Any]) -> List[tuple]:
        """Extract bounding boxes from YOLO result."""
        bboxes = []
        
        # Check for predictions in result
        predictions = yolo_result.get("all_predictions", [])
        
        if not predictions:
            # Try alternative field names
            predictions = yolo_result.get("predictions", [])
        
        for pred in predictions:
            bbox = pred.get("bbox")
            if bbox and len(bbox) == 4:
                # Convert to (x1, y1, x2, y2) format
                x1, y1, x2, y2 = bbox
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return bboxes
    
    def _lookup_nutrition(self, cuisine: str, food_group: str) -> Optional[Dict[str, float]]:
        """Look up average nutrition for a cuisine/food_group combination."""
        registry = self._get_nutrition_registry()
        
        # Try exact lookup first
        nutrition = registry.get_by_cuisine_and_food_group(cuisine, food_group)
        
        if nutrition:
            return nutrition
        
        # Fallback: try with "Other" food group
        nutrition = registry.get_by_cuisine_and_food_group(cuisine, "Other")
        
        if nutrition:
            print(f"[Pipeline] Fallback to {cuisine}/Other for nutrition")
            return nutrition
        
        # Last fallback: search by food group name only
        fuzzy_results = registry.fuzzy_search(food_group, limit=1)
        if fuzzy_results:
            return fuzzy_results[0]
        
        print(f"[Pipeline] No nutrition found for {cuisine}/{food_group}")
        return None
    
    def _calculate_summary(self, detected_foods: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregated nutrition summary."""
        if not detected_foods:
            return {"message": "No foods detected"}
        
        # Find the most confident detection for representative nutrition
        best_detection = max(
            detected_foods, 
            key=lambda x: (x.get("cuisine_confidence", 0) + x.get("food_group_confidence", 0)) / 2
        )
        
        nutrition = best_detection.get("nutrition_per_100g")
        
        if nutrition:
            return {
                "primary_cuisine": best_detection.get("cuisine"),
                "primary_food_group": best_detection.get("food_group"),
                "avg_confidence": (
                    best_detection.get("cuisine_confidence", 0) + 
                    best_detection.get("food_group_confidence", 0)
                ) / 2,
                "nutrition_per_100g": {
                    "calories": nutrition.get("calories", 0),
                    "protein_g": nutrition.get("protein_g", 0),
                    "carbs_g": nutrition.get("carbs_g", 0),
                    "fat_g": nutrition.get("fat_g", 0),
                    "fiber_g": nutrition.get("fiber_g", 0),
                    "density": nutrition.get("density", 1.0)
                },
                "note": "Nutrition values are averages for this food group (per 100g)"
            }
        
        return {
            "primary_cuisine": best_detection.get("cuisine"),
            "primary_food_group": best_detection.get("food_group"),
            "message": "Nutrition data not available for this combination"
        }
    
    def process_crop(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a pre-cropped image (skip YOLO detection).
        
        Use this when you already have a food crop (e.g., from YOLO).
        
        Args:
            image: PIL Image of the food crop
            
        Returns:
            Dict with cuisine, food_group, and nutrition
        """
        clip_classifier = self._get_clip_classifier()
        
        if not clip_classifier.is_available:
            return {"success": False, "error": "CLIP not available"}
        
        # Run hierarchical classification
        clip_result = clip_classifier.classify_hierarchical(image)
        
        if not clip_result.get("success"):
            return {"success": False, "error": clip_result.get("error")}
        
        cuisine = clip_result.get("cuisine")
        food_group = clip_result.get("food_group")
        
        # Lookup nutrition
        nutrition = self._lookup_nutrition(cuisine, food_group)
        
        return {
            "success": True,
            "cuisine": cuisine,
            "cuisine_confidence": clip_result.get("cuisine_confidence"),
            "food_group": food_group,
            "food_group_confidence": clip_result.get("food_group_confidence"),
            "nutrition_per_100g": nutrition
        }


# Global singleton
_pipeline: Optional[HierarchicalFoodPipeline] = None


def get_food_pipeline() -> HierarchicalFoodPipeline:
    """Get or create the global food recognition pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HierarchicalFoodPipeline()
    return _pipeline


def recognize_food_hierarchical(image_path: str) -> Dict[str, Any]:
    """
    Convenience function for full hierarchical food recognition.
    
    Args:
        image_path: Path to the food image
        
    Returns:
        Dict with detected foods and nutrition estimates
    """
    pipeline = get_food_pipeline()
    return pipeline.process_image(image_path)
