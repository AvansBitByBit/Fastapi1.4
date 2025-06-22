@echo off
echo ========================================
echo  Enhanced Waste Detection API v2.0.0
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing/updating dependencies...
pip install -r requirementsgoat.txt

REM Check if .env exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please create a .env file with your API credentials:
    echo.
    echo API_USERNAME=your_username
    echo API_PASSWORD=your_password
    echo API_PASSWORD1=your_alternative_password
    echo.
    echo You can also copy from the original .env file if you have one.
    echo.
    pause
)

REM Check if model exists
if not exist "random_forest_model.pkl" (
    echo.
    echo WARNING: Model file 'random_forest_model.pkl' not found!
    echo Please ensure the model file is in this directory.
    echo.
    pause
)

REM Start the API server
echo.
echo ========================================
echo Starting Enhanced FastAPI server...
echo ========================================
echo.
echo API will be available at:
echo   - Main API: http://localhost:8000
echo   - Documentation: http://localhost:8000/docs
echo   - Alternative docs: http://localhost:8000/redoc
echo   - Health check: http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

uvicorn application:app --reload --host 0.0.0.0 --port 8000
