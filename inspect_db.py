import sqlite3
import json
import os

DB_PATH = "src/auth/health_app.db"

def inspect_db():
    with open("inspect_log.txt", "w") as log:
        log.write("Starting DB inspection...\n")
        if not os.path.exists(DB_PATH):
            log.write(f"Database not found at {DB_PATH}\n")
            return
        log.write(f"Database found at {DB_PATH}. Connecting...\n")

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        log.write("--- Users ---\n")
        cursor.execute("SELECT id, email, name FROM users")
        users = cursor.fetchall()
        for u in users:
            log.write(f"ID: {u['id']}, Email: {u['email']}, Name: {u['name']}\n")

        log.write("\n--- Schema Check (medical_profiles) ---\n")
        cursor.execute("PRAGMA table_info(medical_profiles)")
        cols = cursor.fetchall()
        for c in cols:
            log.write(f"ID: {c[0]}, Name: {c[1]}, Type: {c[2]}\n")

        log.write("\n--- All Medical Profiles ---\n")
        cursor.execute("""
            SELECT mp.id, mp.user_id, mp.created_at, mp.weight_kg, mp.height_cm, u.email
            FROM medical_profiles mp 
            JOIN users u ON mp.user_id = u.id 
            ORDER BY mp.user_id, mp.created_at DESC
        """)
        all_profiles = cursor.fetchall()
        for p in all_profiles:
            log.write(f"User: {p['email']}, Created: {p['created_at']}, W/H: {p['weight_kg']} / {p['height_cm']}\n")

        conn.close()
        log.write("Inspection complete.\n")

if __name__ == "__main__":
    inspect_db()
