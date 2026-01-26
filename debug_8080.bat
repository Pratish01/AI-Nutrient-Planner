@echo off
echo Starting...
set PYTHONPATH=%~dp0
echo CWD: %CD% > launch_log.txt
echo PATH: %PATH% >> launch_log.txt
python --version >> launch_log.txt 2>&1
python src/main.py >> launch_log.txt 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python failed. Trying py... >> launch_log.txt
    py src/main.py >> launch_log.txt 2>&1
)
echo Done. >> launch_log.txt
