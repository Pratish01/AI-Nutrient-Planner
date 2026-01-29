import sqlite3
import os

db_path = os.path.join('data', 'nutrition.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("Checking medical profiles...")
cursor.execute("SELECT COUNT(*) FROM medical_profiles")
print(f"Total Profiles: {cursor.fetchone()[0]}")

print("\nChecking meal logs...")
cursor.execute("SELECT COUNT(*) FROM meal_logs")
print(f"Total Meals: {cursor.fetchone()[0]}")

print("\nLatest Profile:")
cursor.execute("SELECT user_id, conditions, allergens, created_at FROM medical_profiles ORDER BY created_at DESC LIMIT 1")
row = cursor.fetchone()
if row:
    print(row)
else:
    print("No profile found")

conn.close()
