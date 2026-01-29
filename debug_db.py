import sqlite3
import json

def dump_db():
    import os
    db_path = os.path.join('data', 'nutrition.db')
    if not os.path.exists(db_path):
        # try src path as fallback
        db_path = os.path.join('src', 'nutrition.db')
    print(f"Using DB: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("--- MEDICAL PROFILES ---")
    cursor.execute("SELECT * FROM medical_profiles")
    rows = cursor.fetchall()
    for row in rows:
        r = dict(row)
        print(f"User: {r['user_id']} | Conditions: {r['conditions']} | Targets: {r['daily_targets']}")
    
    print("\n--- MEAL LOGS ---")
    cursor.execute("SELECT * FROM meal_logs LIMIT 5")
    rows = cursor.fetchall()
    for row in rows:
        r = dict(row)
        print(f"User: {r['user_id']} | Food: {r['food_name']} | Time: {r['timestamp']}")
    
    conn.close()

if __name__ == "__main__":
    dump_db()
