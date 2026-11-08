@echo off
rem One-time setup: creates .venv and installs Chordfinder's dependencies.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.11 -m venv .venv || python -m venv .venv || goto :error
)
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126 || goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
echo.
echo Done. Start the app with Chordfinder.bat
pause
exit /b 0
:error
echo Installation failed.
pause
exit /b 1
