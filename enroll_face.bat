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

REM Usage:
REM   enroll_face.bat <person_name> [camera_index] [count]

set PERSON_NAME=%1
if "%PERSON_NAME%"=="" (
    set /p PERSON_NAME=Enter person name: 
)

set CAMERA_INDEX=
if not "%2"=="" (
    set CAMERA_INDEX=--camera %2
)

set COUNT_ARG=
if not "%3"=="" (
    set COUNT_ARG=--count %3
)

python enroll_face.py --name "%PERSON_NAME%" %CAMERA_INDEX% %COUNT_ARG%

endlocal
