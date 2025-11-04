@echo off
setlocal

REM Activate venv
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

REM Optional: allow specifying camera index as first argument
set CAMERA_INDEX_ARG=
if not "%1"=="" (
    set CAMERA_INDEX_ARG=--camera %1
)

echo Running recognize_video.py ... (logs will be saved to run_log.txt)
python recognize_video.py %CAMERA_INDEX_ARG% > run_log.txt 2>&1
set EXITCODE=%ERRORLEVEL%
echo ======================================
echo Process exited with code %EXITCODE%
echo Showing run_log.txt below:
echo ======================================
type run_log.txt
echo.
echo (Log also saved to run_log.txt)
pause

endlocal
