@echo off
echo ===================================================
echo             Starting ClearClause...
echo ===================================================
echo.

echo [1/2] Starting Backend Server on port 8000...
start "ClearClause Backend" cmd /k "venv\Scripts\activate.bat && python -m uvicorn backend.main:app --reload --port 8000"

echo [2/2] Starting Frontend Server on port 3000...
start "ClearClause Frontend" cmd /k "cd frontend && python -m http.server 3000"

echo.
echo Both servers are starting up in new windows!
echo.
echo - Backend API:  http://localhost:8000
echo - Frontend UI:  http://localhost:3000/pages/index.html
echo.
echo (To stop the servers later, simply close their command prompt windows.)
echo.
pause
