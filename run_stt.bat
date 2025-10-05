@echo off
REM Batch file to run STT_and_processing.py with immediate output
REM This ensures you see print statements in real-time

REM Set Python to unbuffered mode for immediate output
set PYTHONUNBUFFERED=1

REM Run the Python script with conda environment and keep window open
cmd /k "conda run --name NLP --no-capture-output python "%~dp0example_vibe_checker_ensemble.py" && echo. && echo Script completed! You can close this window now."