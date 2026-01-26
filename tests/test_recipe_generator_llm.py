import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from intelligence.recipe_generator import RecipeGenerator, GeneratedRecipe, CookingMethod
from models.user import UserProfile, HealthCondition
from rules.engine import RuleEngine
from services.llm_service import LLMResponse, OllamaService

class TestRecipeGeneratorLLM(unittest.TestCase):
    def setUp(self):
        self.rule_engine = MagicMock(spec=RuleEngine)
        self.rule_engine.evaluate.return_value = [] # No violations
        self.generator = RecipeGenerator(self.rule_engine)
        self.user = UserProfile(user_id="test", name="Test", age=30, conditions=[], allergens=[])
        
    @patch('intelligence.recipe_generator.get_llm_service')
    def test_llm_fallback(self, mock_get_llm):
        # Setup mock LLM
        mock_llm_service = MagicMock()
        mock_get_llm.return_value = mock_llm_service
        mock_llm_service.is_available = True
        
        # Mock successful JSON response
        mock_response = MagicMock(spec=LLMResponse)
        mock_response.success = True
        mock_response.content = """
        {
            "name": "Strange Mix Special",
            "ingredients": [["unicorn dust", 10, "g"], ["rainbow drops", 20, "ml"]],
            "instructions": ["Mix dust", "Add drops"],
            "cooking_method": "raw",
            "cooking_time_mins": 5,
            "nutrition_per_serving": {
                "calories": 100, "protein_g": 5, "fat_g": 2, "carbs_g": 15, "sugar_g": 10, "sodium_mg": 10, "fiber_g": 1
            },
            "medical_notes": ["Magical notes"]
        }
        """
        mock_llm_service.chat.return_value = mock_response
        
        # Ingredients that definitely won't match a template
        ingredients = ["unicorn dust", "rainbow drops"]
        
        # Call generate
        recipe = self.generator.generate(ingredients, self.user)
        
        # Verify LLM was called
        mock_llm_service.chat.assert_called_once()
        
        # Verify recipe content
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.name, "Strange Mix Special")
        self.assertEqual(recipe.cooking_method, CookingMethod.RAW)
        self.assertEqual(recipe.ingredients[0][0], "unicorn dust")
        self.assertEqual(recipe.total_nutrition['calories'], 100)
        
    @patch('intelligence.recipe_generator.get_llm_service')
    def test_llm_unavailable(self, mock_get_llm):
        # Setup mock LLM as unavailable
        mock_llm_service = MagicMock()
        mock_get_llm.return_value = mock_llm_service
        mock_llm_service.is_available = False
        
        ingredients = ["impossible ingredient"]
        recipe = self.generator.generate(ingredients, self.user)
        
        self.assertIsNone(recipe)

if __name__ == '__main__':
    unittest.main()
