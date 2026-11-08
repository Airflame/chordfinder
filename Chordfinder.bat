@echo off
rem Launches the GUI without a console window.
cd /d "%~dp0"
start "" .venv\Scripts\pythonw.exe -m chordfinder %*
