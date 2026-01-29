import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from auth.database import init_database, DailyLogRepository, MealRepository, MedicalProfileRepository

def verify_analytics():
    print("Initializing Database...")
    init_database()
    
    user_id = "test-user-" + datetime.now().strftime("%H%M%S")
    print(f"User ID: {user_id}")
    
    # 1. Create a dummy medical profile
    print("Creating medical profile...")
    MedicalProfileRepository.create(
        profile_id="prof-1",
        user_id=user_id,
        conditions=["diabetes"],
        allergens=[],
        daily_targets={"calories": 2000, "protein_g": 100, "carbs_g": 200, "fat_g": 60}
    )
    
    # 2. Log some meals
    print("Logging meals...")
    MealRepository.create(user_id, "Apple", {"calories": 95, "protein_g": 0.5, "carbs_g": 25, "fat_g": 0.3})
    MealRepository.create(user_id, "Chicken Salad", {"calories": 350, "protein_g": 30, "carbs_g": 10, "fat_g": 20})
    
    # 3. Fetch today's stats
    today_str = datetime.now().strftime("%Y-%m-%d")
    stats = DailyLogRepository.get_or_create(user_id, today_str)
    print(f"Today's Stats: {stats}")
    
    # Verify values
    if stats['calories_consumed'] == 445:
        print("✅ Calorie aggregation successful")
    else:
        print(f"❌ Calorie aggregation failed: expected 445, got {stats['calories_consumed']}")
        
    # 4. Verify Analytics Summary Logic (Portion of main.py logic)
    print("Verifying summary logic...")
    weekly_logs = []
    end_date = datetime.now()
    for i in range(7):
        date_str = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
        log = DailyLogRepository.get_or_create(user_id, date_str)
        weekly_logs.append(log)
    
    total_actual_cals = sum(l.get("calories_consumed", 0) for l in weekly_logs)
    total_target_cals = sum(l.get("calories_target", 2000) for l in weekly_logs)
    
    print(f"Weekly Actual: {total_actual_cals}, Weekly Target: {total_target_cals}")
    
    if total_actual_cals == 445:
        print("✅ Weekly aggregation successful")
    else:
        print("❌ Weekly aggregation failed")

if __name__ == "__main__":
    verify_analytics()
