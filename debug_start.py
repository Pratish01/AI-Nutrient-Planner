import sys
import traceback
import os

log_file = "startup_log.txt"

def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

# Clear log
with open(log_file, "w", encoding="utf-8") as f:
    f.write("DEBUG START\n")

try:
    log(f"CWD: {os.getcwd()}")
    log(f"Python: {sys.executable}")
    
    # Try import
    log("Importing uvicorn...")
    import uvicorn
    
    log("Importing app form src.main...")
    # Add CWD to path to ensure src is found
    sys.path.insert(0, os.getcwd())
    
    from src.main import app
    log("App imported successfully.")
    
    # Check routes in app
    log("Checking routes in imported app object:")
    has_api_login = False
    for r in app.routes:
        path = getattr(r, 'path', str(r))
        log(f"  - {path}")
        if '/api/auth/login' in path:
            has_api_login = True
            
    log(f"Found /api/auth/login: {has_api_login}")
            
    log("Starting Uvicorn on 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
    
except Exception as e:
    log("CRASHED!")
    log(traceback.format_exc())
