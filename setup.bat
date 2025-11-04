@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Setup virtual environment and install dependencies

REM Determine Python command
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    set PY_CMD=py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
        set PY_CMD=python
    ) else (
        echo Python not found on PATH. Please install Python 3 and try again.
        pause
        exit /b 1
    )
)

REM Create venv if it doesn't exist
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment in .venv ...
    %PY_CMD% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment detected: .venv
)

REM Upgrade pip and install requirements
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Python dependencies.
    pause
    exit /b 1
)

echo.
echo Setup complete.
echo Next steps:
echo  1) Set your Telegram bot credentials in config.py or environment variables:
echo     - TELEGRAM_BOT_TOKEN
echo     - TELEGRAM_ADMIN_CHAT_ID

echo  2) To run the project, use: run_project.bat

echo  3) To enroll faces into the dataset, use: enroll_face.bat

echo.
pause
endlocal
