@echo off
rem CaseFlow: start the engine (port 8600) and the web app (port 5190), then open the browser.
rem Python: uses .venv in this folder if present, otherwise the parent folder's .venv.
cd /d %~dp0
set PY=.venv\Scripts\python.exe
if not exist %PY% set PY=..\.venv\Scripts\python.exe
start "CaseFlow engine" cmd /k "%PY% -m uvicorn server.app:app --host 127.0.0.1 --port 8600"
if not exist web\node_modules (cd web && call npm install && cd ..)
start "CaseFlow web" cmd /k "cd web && npm run dev"
timeout /t 6 >nul
start http://localhost:5190
