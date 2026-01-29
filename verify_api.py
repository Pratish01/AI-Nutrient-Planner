
import requests
import json
import sys

# Configuration
USER_EMAIL = "demo@example.com"
USER_PASSWORD = "password123"

# Direct DB Seed
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from auth.database import MedicalProfileRepository, init_database
import uuid

def run_verification():
    print(f"--- STARTING API VERIFICATION ---")
    
    # Try multiple targets
    target_url = None
    for port in [8081, 8000]:
        url = f"http://127.0.0.1:{port}"
        print(f"Checking {url}...")
        try:
            requests.get(f"{url}/health", timeout=1)
            target_url = url
            print(f"✅ Found server at {target_url}")
            break
        except:
            pass
            
    if not target_url:
        print("❌ Could not find running server on port 8081 or 8000 (127.0.0.1)")
        return

    BASE_URL = target_url
    
    # 1. Login
    print(f"\n[1] Attempting Login ({USER_EMAIL})...")
    try:
        login_resp = requests.post(f"{BASE_URL}/auth/login", json={"email": USER_EMAIL, "password": USER_PASSWORD})
        
        # Auto-register if user missing
        if login_resp.status_code == 401:
            print("⚠️  Login failed (401). Attempting to REGISTER demo user...")
            reg_resp = requests.post(f"{BASE_URL}/auth/register", json={
                "email": USER_EMAIL, 
                "password": USER_PASSWORD,
                "name": "Demo User"
            })
            if reg_resp.status_code == 200:
                print("✅ Registration Success! Retrying login...")
                login_resp = requests.post(f"{BASE_URL}/auth/login", json={"email": USER_EMAIL, "password": USER_PASSWORD})
            else:
                print(f"❌ Registration Failed: {reg_resp.text}")
                return

        if login_resp.status_code != 200:
            print(f"❌ Login Failed: {login_resp.status_code} - {login_resp.text}")
            return
        
        data = login_resp.json()
        if login_resp.status_code != 200:
            print(f"❌ Login Failed: {login_resp.status_code} - {login_resp.text}")
            return
        
        data = login_resp.json()
        token = data.get("token")
        user_id = data.get("user_id")
        
        if not token:
            print("❌ Login success but NO TOKEN returned.")
            print(data)
            return

        print(f"✅ Login Success! Token obtained.")
        headers = {"Authorization": f"Bearer {token}"}
        
        # 1.5 SEED PROFILE IF MISSING (The final fix)
        print(f"\n[1.5] Checking for Medical Profile in DB...")
        
        # Helper to force-clean bad data
        def clean_slate():
             with init_database() as conn: # This is a context manager but init_database returns None?
                 # Wait, init_database initializes tables. We need get_connection
                 pass
        
        # Use a fresh connection for direct SQL manip if needed, but let's try repo first
        existing_profile = MedicalProfileRepository.get_by_user_id(user_id)
        
        # Check if profile is "empty" (missing weight) even if it exists
        needs_seed = False
        if not existing_profile:
            print("⚠️  No profile found.")
            needs_seed = True
        elif not existing_profile.get("weight_kg"):
             print("⚠️  Profile exists but HAS NO WEIGHT data.")
             needs_seed = True
             
        if needs_seed:
            print("SEEDING DEFAULT DATA (75kg, 180cm)...")
            try:
                # 1. Nuke old partial profile to avoid conflicts
                from auth.database import get_connection
                with get_connection() as conn:
                    conn.execute("DELETE FROM medical_profiles WHERE user_id = ?", (user_id,))
                    conn.commit()
                
                # 2. Create fresh
                MedicalProfileRepository.create(
                    profile_id=str(uuid.uuid4()),
                    user_id=user_id,
                    conditions=[],
                    allergens=[],
                    medications=[],
                    daily_targets={"calories": 2000, "protein_g": 150},
                    raw_ocr_text="Auto-Seeded via Verify Script",
                    source_file="auto_seed.txt",
                    age=30,
                    gender="male",
                    weight_kg=75.0,
                    height_cm=180.0,
                    activity_level="moderately_active",
                    fitness_goal="maintain_weight"
                )
                print("✅ Profile SEEDED (75kg, 180cm)!")
            except Exception as e:
                print(f"❌ Seed Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
             print(f"✅ Profile Valid. Weight: {existing_profile.get('weight_kg')}kg")
        
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # 2. Get Profile
    print(f"\n[2] Fetching Medical Profile...")
    try:
        prof_resp = requests.get(f"{BASE_URL}/api/profile", headers=headers)
        if prof_resp.status_code == 200:
            prof_data = prof_resp.json()
            bio = prof_data.get("bio_metrics", {})
            print(f"✅ Profile Response: 200 OK")
            print(f"   > Name: {prof_data.get('name')}")
            print(f"   > Age: {bio.get('age')}")
            print(f"   > Weight: {bio.get('weight_kg')} kg")
            print(f"   > Height: {bio.get('height_cm')} cm")
            print(f"   > BMI: {bio.get('bmi')}")
        else:
            print(f"❌ Profile Fetch Failed: {prof_resp.status_code}")
            print(prof_resp.text)
    except Exception as e:
        print(f"❌ Profile Error: {e}")

    # 3. Get Daily Stats
    print(f"\n[3] Fetching Daily Stats (Today)...")
    try:
        stats_resp = requests.get(f"{BASE_URL}/api/daily-stats", headers=headers)
        if stats_resp.status_code == 200:
            stats_data = stats_resp.json()
            print(f"✅ Stats Response: 200 OK")
            print(f"   > Calories: {stats_data.get('calories_consumed')} / {stats_data.get('calories_target')}")
            print(f"   > Protein: {stats_data.get('protein_g')}")
            print(f"   > Water: {stats_data.get('water_cups')} cups")
        else:
            print(f"❌ Stats Fetch Failed: {stats_resp.status_code}")
            print(stats_resp.text)
    except Exception as e:
        print(f"❌ Stats Error: {e}")

    print(f"\n--- VERIFICATION COMPLETE ---")

if __name__ == "__main__":
    run_verification()
