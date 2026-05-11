@echo off
title ASLOR Proxy
cd /d "%~dp0"

echo.
echo   ASLOR -- Android Studio LLM OpenAI Reasoning Proxy
echo   ===================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python is not on your PATH.
    echo   Install Python 3.11+ from https://python.org and add it to PATH.
    pause
    exit /b 1
)

:: Check for config.yaml
if not exist "config.yaml" (
    echo   [INFO]  config.yaml not found -- copying config.example.yaml
    copy config.example.yaml config.yaml >nul
)

:: Launch
echo   Starting proxy on http://127.0.0.1:3001
echo   Dashboard at http://127.0.0.1:3001/dashboard
echo   Auto-reload is enabled for code, YAML, and .env changes
echo   Press Ctrl+C to stop.
echo.
if "%ASLOR_RELOAD%"=="" set ASLOR_RELOAD=1
python -m aslor.main

pause
