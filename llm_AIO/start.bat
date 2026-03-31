@echo off
REM =========================================
REM Playground AI Server - 一键启动脚本
REM 适用于: Windows 服务器
REM =========================================
setlocal

echo =====================================
echo   Playground AI Gateway - Starting
echo =====================================

REM --- 配置区域 (请根据服务器环境修改) ---
set PYTHON_PATH=D:\Anaconda3\envs\lf\python.exe
set HOST=0.0.0.0
set PORT=8000

REM --- 清除代理 (防止影响连接阿里云) ---
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

REM --- 切换到脚本所在目录 ---
cd /d %~dp0

REM --- 检查 Python ---
%PYTHON_PATH% --version > nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found! Please install Python or update PYTHON_PATH.
    pause
    exit /b 1
)

REM --- 检查 .env 文件 ---
if not exist ".env" (
    echo [WARN] .env file not found! Creating from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo [INFO] Please edit .env and add your API keys.
        notepad .env
    ) else (
        echo [ERROR] .env.example not found!
    )
)

REM --- 启动服务器 ---
echo [INFO] Starting server on http://%HOST%:%PORT%
echo [INFO] Press Ctrl+C to stop.
echo.

%PYTHON_PATH% -m uvicorn app.main:app --host %HOST% --port %PORT%

pause
