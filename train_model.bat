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

echo ==============================================
echo Extracting face embeddings from dataset/ ...
echo ==============================================
python extract_embeddings.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during embeddings extraction.
    pause
    exit /b 1
)

echo ===============================
echo Training recognition model ...
echo ===============================
python train_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Error during model training.
    pause
    exit /b 1
)

echo.
echo Training complete. Files written to output/ :
echo  - embeddings.pickle
echo  - recognizer (model)
echo  - le.pickle (label encoder)
echo.
pause
endlocal
