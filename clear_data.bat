@echo off
setlocal ENABLEDELAYEDEXPANSION

echo This will DELETE existing face data and trained models.
echo - All images under dataset\
- All clips under output\unknown_clips\
- output\embeddings.pickle
- output\recognizer and output\recognizer.pickle
- output\le.pickle
echo.
set /p CONFIRM=Are you sure you want to proceed? (Y/N): 
if /I not "%CONFIRM%"=="Y" (
    echo Aborted.
    goto :eof
)

REM Ensure directories exist
if not exist dataset mkdir dataset
if not exist output mkdir output
if not exist output\unknown_clips mkdir output\unknown_clips

REM Delete dataset images
if exist dataset (
    echo Deleting dataset images...
    for /d %%D in ("dataset\*") do (
        echo Removing folder: %%D
        rmdir /s /q "%%D"
    )
)

REM Delete unknown clips
if exist output\unknown_clips (
    echo Deleting unknown clips...
    del /q "output\unknown_clips\*" >nul 2>&1
)

REM Delete model files
if exist output\embeddings.pickle del /q output\embeddings.pickle
if exist output\recognizer del /q output\recognizer
if exist output\recognizer.pickle del /q output\recognizer.pickle
if exist output\le.pickle del /q output\le.pickle

echo.
echo Cleanup complete.
echo Next steps:
echo  1) Enroll faces:   enroll_face.bat <name> [camera_index] [count]
echo  2) Train model:    train_model.bat
echo  3) Run project:    run_project.bat [camera_index]

echo.
pause
endlocal
