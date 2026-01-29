import ast
import sys
import os

files_to_check = [
    r"c:\Users\hp\AI Nutrition\src\main.py",
    r"c:\Users\hp\AI Nutrition\src\auth\database.py"
]

print("--- STARTING SYNTAX CHECK ---")
has_error = False

for file_path in files_to_check:
    print(f"Checking {os.path.basename(file_path)}...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print(f"✅ Syntax OK")
    except SyntaxError as e:
        has_error = True
        print(f"❌ SYNTAX ERROR in {os.path.basename(file_path)}:")
        print(f"   Line {e.lineno}: {e.text.strip() if e.text else ''}")
        print(f"   Message: {e.msg}")
    except Exception as e:
        has_error = True
        print(f"❌ Error reading/parsing: {e}")

if has_error:
    print("\n⚠️  FIX ERRORS BEFORE STARTING SERVER  ⚠️")
    sys.exit(1)
else:
    print("\nAll files look good! Server *should* start.")
    print("If it still fails, check for runtime errors (imports/ports).")
