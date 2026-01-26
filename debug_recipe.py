from intelligence.recipe_generator import RecipeGenerator, GeneratedRecipe, RecipeTemplate, CookingMethod
from models.user import UserProfile, HealthCondition
from rules.engine import RuleEngine
import json

# Setup
rule_engine = RuleEngine()
generator = RecipeGenerator(rule_engine)
user = UserProfile(user_id="test", name="Test", age=30, conditions=[], allergens=[])

# 1. Test Template Generation
print("--- TEMPLATE GENERATION ---")
recipe = generator.generate(["chicken", "broccoli", "rice"], user)
if recipe:
    print(json.dumps(recipe.to_dict(), indent=2))
else:
    print("No recipe generated.")

# 2. Test LLM Fallback (mocked by passing odd ingredients)
# Note: This will actually call LLM if configured, or fail gracefully. 
# We just want to see the to_dict structure.
