
import sys
import os
from unittest.mock import MagicMock

# Attempt to mock transformers if missing, to allow app import
sys.modules['transformers'] = MagicMock()
sys.modules['transformers.SiglipImageProcessor'] = MagicMock()
sys.modules['transformers.SiglipModel'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from fastapi.testclient import TestClient

# Mock heavy services BEFORE importing main
sys.modules['services.continental_retrieval'] = MagicMock()
sys.modules['services.nutrition_registry'] = MagicMock()
sys.modules['services.llm_service'] = MagicMock()
sys.modules['services.rag_service'] = MagicMock()
sys.modules['analytics.weight_forecaster'] = MagicMock()

# Mock database module to prevent connection attempts
mock_auth_db = MagicMock()
sys.modules['auth.database'] = mock_auth_db

# Mock specific functions called at module level
sys.modules['services.continental_retrieval'].get_continental_retrieval_system = MagicMock()
sys.modules['services.nutrition_registry'].get_nutrition_registry = MagicMock()
sys.modules['analytics.weight_forecaster'].get_weight_forecaster = MagicMock()

from main import app, get_current_user
from auth.auth_service import auth_service
# We mocked auth.database, so we must access the repo via the mock or re-assign
MedicalProfileRepository = MagicMock()
# Assign it back to the mock module so main.py finds it if it imports it (though main imports specific names)
# main.py does: from auth.database import (..., MedicalProfileRepository, ...)
# So we need to set attributes on the mock module
mock_auth_db.UserRepository = MagicMock()
mock_auth_db.MedicalProfileRepository = MedicalProfileRepository
mock_auth_db.UploadRepository = MagicMock()
mock_auth_db.MealRepository = MagicMock()
mock_auth_db.DailyLogRepository = MagicMock()
mock_auth_db.init_database = MagicMock()

# IMPORTANT: Since main.py ALREADY imported logic before we could patch if it was top level...
# But we are importing main AFTER patching sys.modules, so main should see the mocks.


def mock_get_current_user():
    return {"sub": "test-user-id"}

# Override dependency
app.dependency_overrides[get_current_user] = mock_get_current_user

# Mock database response
mock_profile = {
    "user_id": "test-user-id",
    "conditions": ["diabetes"],
    "allergens": ["peanuts"],
    "daily_targets": {
        "calories": 2000,
        "sugar_level": "100 mg/dL",
        "cholesterol": "180 mg/dL"
    },
    "age": 30,
    "weight_kg": 70,
    "height_cm": 175,
    "gender": "male",
    "fitness_goal": "maintenance",
    "activity_level": "moderately_active"
}

MedicalProfileRepository.get_by_user_id = MagicMock(return_value=mock_profile)

client = TestClient(app)

def test_profile_endpoint():
    print("Testing /api/profile endpoint...")
    response = client.get("/api/profile")
    
    if response.status_code != 200:
        print(f"FAILED: Status code {response.status_code}")
        print(response.json())
        sys.exit(1)
        
    data = response.json()
    print("Response received.")
    
    # Verify allergies mapping
    if "allergies" not in data:
        print("FAILED: 'allergies' key missing")
        sys.exit(1)
        
    if data["allergies"] != ["peanuts"]:
        print(f"FAILED: 'allergies' content mismatch. Got {data['allergies']}")
        sys.exit(1)
        
    print("SUCCESS: 'allergies' field present and correct.")
    
    # Verify vitals extraction
    if "vitals" not in data:
        print("FAILED: 'vitals' key missing")
        sys.exit(1)
    
    vitals = data["vitals"]
    if vitals.get("glucose_level") != "100 mg/dL":
         print(f"FAILED: glucose_level mismatch. Got {vitals.get('glucose_level')}")
         sys.exit(1)
         
    if vitals.get("cholesterol") != "180 mg/dL":
         print(f"FAILED: cholesterol mismatch. Got {vitals.get('cholesterol')}")
         sys.exit(1)

    print("SUCCESS: 'vitals' field present and correct.")
    
    # Verify bio_metrics BMI
    if "bio_metrics" not in data:
        print("FAILED: 'bio_metrics' key missing")
        sys.exit(1)
    
    bmi = data["bio_metrics"].get("bmi")
    # 70 / (1.75^2) = 22.857 -> 22.9
    if bmi != 22.9:
        print(f"FAILED: BMI calculation incorrect. Got {bmi}, expected 22.9")
        sys.exit(1)
        
    print("SUCCESS: BMI calculated correctly.")
    
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    test_profile_endpoint()
