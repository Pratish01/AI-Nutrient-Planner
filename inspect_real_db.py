import sqlite3
import json
import os

DB_PATH = "data/nutrition.db"

def inspect_db():
    with open("inspect_real_log.txt", "w") as log:
        if not os.path.exists(DB_PATH):
            log.write(f"Database not found at {DB_PATH}\n")
            return

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            log.write("--- Medical Profiles (Latest 10) ---\n")
            cursor.execute("""
                SELECT mp.id, mp.user_id, mp.age, mp.weight_kg, mp.height_cm, mp.created_at, u.email, u.name
                FROM medical_profiles mp 
                JOIN users u ON mp.user_id = u.id 
                ORDER BY mp.created_at DESC 
                LIMIT 10
            """)
            profiles = cursor.fetchall()
            for p in profiles:
                log.write(f"User: {p['email']} ({p['name']}), Created: {p['created_at']}\n")
                log.write(f"  Age: {p['age']}, Weight: {p['weight_kg']}, Height: {p['height_cm']}\n")
                log.write("-" * 20 + "\n")
        except Exception as e:
            log.write(f"Error: {e}\n")

        conn.close()

if __name__ == "__main__":
    inspect_db()
