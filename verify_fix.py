import requests
import json
import time

BASE_URL = "http://localhost:8081"
EMAIL = f"test_{int(time.time())}@example.com"
PASSWORD = "password123"

def test_bmi_persistence():
    print("--- Starting BMI Persistence Test ---")
    
    # 1. Register
    reg_resp = requests.post(f"{BASE_URL}/auth/register", json={
        "email": EMAIL, "password": PASSWORD, "name": "Test User"
    })
    token = reg_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"User registered. Token: {token[:10]}...")

    # 2. Complete Profile (Set weight/height)
    setup_resp = requests.post(f"{BASE_URL}/api/user/complete-profile", headers=headers, json={
        "age": 30,
        "gender": "male",
        "weight_kg": 80.0,
        "height_cm": 180.0,
        "activity_level": "moderately_active",
        "fitness_goal": "maintenance"
    })
    print(f"Profile setup: {setup_resp.status_code}")

    # 3. Check Initial Profile & BMI
    prof_resp = requests.get(f"{BASE_URL}/api/profile", headers=headers)
    prof_data = prof_resp.json()
    initial_bmi = prof_data.get("bio_metrics", {}).get("bmi")
    print(f"Initial BMI: {initial_bmi}")
    assert initial_bmi == 24.7

    # 4. Mock a Medical Report Upload (Missing biometrics)
    # We'll use a small text file as a mock report
    try:
        with open("mock_report.txt", "w") as f:
            f.write("Patient has Diabetes. Blood pressure is 130/85. Cholesterol 210.")
        
        with open("mock_report.txt", "rb") as f:
            upload_resp = requests.post(
                f"{BASE_URL}/api/medical-report/upload",
                headers=headers,
                files={"file": ("report.txt", f, "text/plain")}
            )
        print(f"Report uploaded: {upload_resp.status_code}")
    finally:
        import os
        if os.path.exists("mock_report.txt"):
            os.remove("mock_report.txt")

    # 5. Check Profile & BMI again
    prof_resp = requests.get(f"{BASE_URL}/api/profile", headers=headers)
    prof_data = prof_resp.json()
    final_bmi = prof_data.get("bio_metrics", {}).get("bmi")
    final_weight = prof_data.get("bio_metrics", {}).get("weight_kg")
    
    print(f"Final BMI: {final_bmi}")
    print(f"Final Weight: {final_weight}")
    
    assert final_weight == 80.0
    assert final_bmi == 24.7
    print("--- SUCCESS: BMI Persisted and Correctly Calculated ---")

if __name__ == "__main__":
    try:
        test_bmi_persistence()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
