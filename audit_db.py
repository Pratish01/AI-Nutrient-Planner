import sqlite3
import os

DB_PATH = "data/nutrition.db"

def audit():
    if not os.path.exists(DB_PATH):
        with open("audit_results.txt", "w") as f:
            f.write("DB NOT FOUND")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    with open("audit_results.txt", "w") as f:
        f.write("=== MEDICAL PROFILES AUDIT ===\n")
        cursor.execute("SELECT * FROM medical_profiles ORDER BY created_at DESC")
        rows = cursor.fetchall()
        for row in rows:
            r = dict(row)
            f.write(f"ID: {r['id'][:8]}... | User: {r['user_id'][:8]}... | Created: {r['created_at']}\n")
            f.write(f"  Biometrics: Age={r.get('age')}, W={r.get('weight_kg')}, H={r.get('height_cm')}\n")
            f.write(f"  Vitals (JSON snippet): {r.get('daily_targets', '{}')[:50]}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\n=== USERS ===\n")
        cursor.execute("SELECT id, email, name FROM users")
        u_rows = cursor.fetchall()
        for u in u_rows:
            f.write(f"ID: {u['id'][:8]}... | Email: {u['email']} | Name: {u['name']}\n")

    conn.close()

if __name__ == "__main__":
    audit()
