"""
Food Estimation Service

Estimates portion sizes from segmentation masks and calculates scaled nutrition values.
Uses density data from the nutrition database for volume-to-weight conversion.
"""

import os
import csv
import math
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from services.nutrition_registry import get_nutrition_registry

# =============================================================================
# CONFIGURATION
# =============================================================================

# Estimation constants
# Assuming average food item photographed from ~30cm distance
# These are rough approximations and can be calibrated
PIXELS_PER_CM = 50  # Approximate pixels per cm at standard distance
DEFAULT_FOOD_HEIGHT_CM = 2.5  # Average food item height
DEFAULT_DENSITY = 1.0  # g/cm³ (water density as default)
STANDARD_SERVING_GRAMS = 100  # All nutrition values are per 100g


class FoodEstimationService:
    """
    Service for estimating food portions from image segmentation data.
    
    Uses:
    - Segmentation mask area to estimate food surface area
    - Density data to convert volume to weight
    - Scaling factors to adjust nutrition values
    """
    
    def __init__(self):
        """Initialize the estimation service."""
        self.registry = get_nutrition_registry()
        self._density_data: Dict[str, float] = {}
        self._nutrition_data: Dict[str, Dict[str, float]] = {}
        self._load_data()
    
    def _load_data(self):
        """Load density and nutrition mapping from registry."""
        try:
            all_food = self.registry.get_all()
            for row in all_food:
                name = row['name'].lower()
                # Store density from standardized registry entry
                self._density_data[name] = row.get('density', DEFAULT_DENSITY)
                
                # Store full nutrition data
                self._nutrition_data[name] = {
                    "calories": row['calories'],
                    "protein_g": row['protein_g'],
                    "carbs_g": row['carbs_g'],
                    "fat_g": row['fat_g'],
                    "sugar_g": row['sugar_g'],
                    "fiber_g": row['fiber_g'],
                    "sodium_mg": row['sodium_mg'],
                }
            
            print(f"[FoodEstimation] Loaded data for {len(self._nutrition_data)} foods from Registry")
                
        except Exception as e:
            print(f"[FoodEstimation] Error loading data from Registry: {e}")

    
    def get_density(self, food_name: str) -> float:
        """
        Get density for a food item.
        
        Args:
            food_name: Name of the food
            
        Returns:
            Density in g/cm³
        """
        return self._density_data.get(food_name.lower(), DEFAULT_DENSITY)
    
    def estimate_portion_from_mask(
        self,
        mask_area_pixels: int,
        food_name: str,
        image_width: int = 640,
        image_height: int = 480,
        depth_estimate_cm: float = DEFAULT_FOOD_HEIGHT_CM
    ) -> Dict[str, Any]:
        """
        Estimate portion size from segmentation mask area.
        
        Args:
            mask_area_pixels: Number of pixels in the segmentation mask
            food_name: Name of the detected food
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            depth_estimate_cm: Estimated depth/height of the food in cm
            
        Returns:
            Dict with estimated grams, volume, and confidence
        """
        try:
            # Convert pixel area to cm²
            # This is a rough approximation based on assumed camera distance
            pixels_per_cm_squared = PIXELS_PER_CM ** 2
            surface_area_cm2 = mask_area_pixels / pixels_per_cm_squared
            
            # Estimate volume (surface area × depth)
            # For flat foods (like roti), use smaller depth
            # For round foods (like fruits), use larger depth
            volume_cm3 = surface_area_cm2 * depth_estimate_cm
            
            # Get density for this food
            density = self.get_density(food_name)
            
            # Calculate weight
            weight_grams = volume_cm3 * density
            
            # Calculate confidence based on assumptions
            # Lower confidence for very small or very large estimates
            if weight_grams < 10:
                confidence = 0.3
            elif weight_grams > 1000:
                confidence = 0.4
            else:
                confidence = 0.7
            
            # Calculate percentage of image covered
            image_area = image_width * image_height
            coverage_percent = (mask_area_pixels / image_area) * 100
            
            return {
                "success": True,
                "estimated_grams": round(weight_grams, 1),
                "estimated_volume_cm3": round(volume_cm3, 1),
                "surface_area_cm2": round(surface_area_cm2, 1),
                "density_used": density,
                "depth_estimate_cm": depth_estimate_cm,
                "image_coverage_percent": round(coverage_percent, 1),
                "confidence": confidence,
                "estimation_method": "mask_area_volume"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "estimated_grams": STANDARD_SERVING_GRAMS,
                "confidence": 0.1
            }
    
    def estimate_portion_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        food_name: str,
        image_width: int = 640,
        image_height: int = 480
    ) -> Dict[str, Any]:
        """
        Estimate portion size from bounding box (when mask not available).
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            food_name: Name of the detected food
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            
        Returns:
            Dict with estimated grams
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Assume ~60% of bounding box is actual food
        estimated_mask_area = bbox_area * 0.6
        
        result = self.estimate_portion_from_mask(
            int(estimated_mask_area),
            food_name,
            image_width,
            image_height
        )
        
        # Lower confidence for bbox-based estimation
        if result.get("success"):
            result["confidence"] *= 0.7
            result["estimation_method"] = "bbox_approximation"
        
        return result
    
    def scale_nutrition(
        self,
        nutrition: Dict[str, float],
        estimated_grams: float,
        per_serving_grams: float = STANDARD_SERVING_GRAMS
    ) -> Dict[str, float]:
        """
        Scale nutrition values based on estimated portion size.
        
        Args:
            nutrition: Nutrition values per serving
            estimated_grams: Estimated portion size in grams
            per_serving_grams: The serving size the nutrition values are based on
            
        Returns:
            Scaled nutrition values
        """
        scale_factor = estimated_grams / per_serving_grams
        
        scaled = {}
        for key, value in nutrition.items():
            if isinstance(value, (int, float)):
                scaled[key] = round(value * scale_factor, 2)
            else:
                scaled[key] = value
        
        # Add metadata
        scaled["_estimated_grams"] = round(estimated_grams, 1)
        scaled["_scale_factor"] = round(scale_factor, 2)
        scaled["_based_on_serving_grams"] = per_serving_grams
        
        return scaled
    
    def get_nutrition(self, food_name: str) -> Optional[Dict[str, float]]:
        """
        Get nutrition data for a food item.
        
        Args:
            food_name: Name of the food
            
        Returns:
            Nutrition dict or None if not found
        """
        food_lower = food_name.lower()
        
        # Exact match first
        if food_lower in self._nutrition_data:
            return self._nutrition_data[food_lower]
        
        # Fuzzy match - find foods containing this name
        for name, nutrition in self._nutrition_data.items():
            if food_lower in name or name in food_lower:
                print(f"[FoodEstimation] Fuzzy matched '{food_name}' to '{name}'")
                return nutrition
        
        # Fallback defaults for common YOLO COCO food classes
        COCO_FOOD_DEFAULTS = {
            "pizza": {"calories": 266, "protein_g": 11, "carbs_g": 33, "fat_g": 10, "sugar_g": 3.6, "fiber_g": 2.3, "sodium_mg": 598},
            "hot dog": {"calories": 290, "protein_g": 10.4, "carbs_g": 24, "fat_g": 18, "sugar_g": 4, "fiber_g": 0.8, "sodium_mg": 810},
            "sandwich": {"calories": 252, "protein_g": 10, "carbs_g": 32, "fat_g": 9, "sugar_g": 5, "fiber_g": 2, "sodium_mg": 500},
            "cake": {"calories": 257, "protein_g": 4, "carbs_g": 36, "fat_g": 11, "sugar_g": 23, "fiber_g": 0.5, "sodium_mg": 230},
            "donut": {"calories": 452, "protein_g": 5, "carbs_g": 51, "fat_g": 25, "sugar_g": 22, "fiber_g": 1.7, "sodium_mg": 326},
            "carrot": {"calories": 41, "protein_g": 0.9, "carbs_g": 9.6, "fat_g": 0.2, "sugar_g": 4.7, "fiber_g": 2.8, "sodium_mg": 69},
            "apple": {"calories": 52, "protein_g": 0.3, "carbs_g": 14, "fat_g": 0.2, "sugar_g": 10, "fiber_g": 2.4, "sodium_mg": 1},
            "orange": {"calories": 47, "protein_g": 0.9, "carbs_g": 12, "fat_g": 0.1, "sugar_g": 9, "fiber_g": 2.4, "sodium_mg": 0},
            "banana": {"calories": 89, "protein_g": 1.1, "carbs_g": 23, "fat_g": 0.3, "sugar_g": 12, "fiber_g": 2.6, "sodium_mg": 1},
            "broccoli": {"calories": 34, "protein_g": 2.8, "carbs_g": 7, "fat_g": 0.4, "sugar_g": 1.7, "fiber_g": 2.6, "sodium_mg": 33},
        }
        
        if food_lower in COCO_FOOD_DEFAULTS:
            print(f"[FoodEstimation] Using default nutrition for COCO food '{food_name}'")
            return COCO_FOOD_DEFAULTS[food_lower]
        
        print(f"[FoodEstimation] No nutrition found for '{food_name}'")
        return None
    
    def get_depth_estimate(self, food_name: str) -> float:
        """
        Get estimated depth/height for a food type.
        
        Different foods have different typical heights:
        - Flat foods (roti, dosa): ~0.5 cm
        - Bowls (curry, dal): ~3-4 cm
        - Tall foods (burger): ~5-8 cm
        """
        food_lower = food_name.lower()
        
        # Flat foods
        flat_keywords = ['roti', 'chapati', 'paratha', 'dosa', 'uttapam', 'naan', 'bread']
        for kw in flat_keywords:
            if kw in food_lower:
                return 0.5
        
        # Bowl foods
        bowl_keywords = ['curry', 'dal', 'soup', 'kheer', 'raita', 'sabzi', 'stew']
        for kw in bowl_keywords:
            if kw in food_lower:
                return 3.5
        
        # Tall foods
        tall_keywords = ['burger', 'sandwich', 'cake', 'pastry', 'samosa']
        for kw in tall_keywords:
            if kw in food_lower:
                return 5.0
        
        # Rice/biryani
        if 'rice' in food_lower or 'biryani' in food_lower or 'pulao' in food_lower:
            return 3.0
        
        # Default
        return DEFAULT_FOOD_HEIGHT_CM


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_estimation_service: Optional[FoodEstimationService] = None


def get_estimation_service() -> FoodEstimationService:
    """Get or create the global estimation service instance."""
    global _estimation_service
    if _estimation_service is None:
        _estimation_service = FoodEstimationService()
    return _estimation_service


def estimate_portion(
    mask_area_pixels: int,
    food_name: str,
    image_width: int = 640,
    image_height: int = 480
) -> Dict[str, Any]:
    """
    Convenience function to estimate portion size.
    
    Args:
        mask_area_pixels: Number of pixels in the segmentation mask
        food_name: Name of the detected food
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        Estimation results
    """
    service = get_estimation_service()
    depth = service.get_depth_estimate(food_name)
    return service.estimate_portion_from_mask(
        mask_area_pixels, food_name, image_width, image_height, depth
    )
