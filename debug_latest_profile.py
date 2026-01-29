import sqlite3
import os
import json

db_path = os.path.join('data', 'nutrition.db')
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

cursor.execute("SELECT * FROM medical_profiles ORDER BY created_at DESC LIMIT 1")
row = cursor.fetchone()
if row:
    r = dict(row)
    print(f"User ID: {r['user_id']}")
    print(f"Conditions: {r['conditions']}")
    print(f"Allergens: {r['allergens']}")
    print(f"Daily Targets: {r['daily_targets']}")
    print(f"Biometrics: Weight={r['weight_kg']}, Height={r['height_cm']}")
else:
    print("No profiles found")

conn.close()
