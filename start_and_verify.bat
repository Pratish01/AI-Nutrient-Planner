@echo off
echo [0/3] Cleaning up old processes...
taskkill /F /IM python.exe >nul 2>&1

echo [1/3] Launching Server in new window...
start "AI Nutrition Server" cmd /k "python src/main.py || pause"

echo [2/3] Waiting 5 seconds for server to boot...
timeout /t 5 /nobreak >nul

echo [3/3] Running Verification...
python verify_api.py

echo.
echo If you see "Found server", the fix is working!
pause
