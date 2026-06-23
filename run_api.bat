@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv\Scripts\python.exe
    echo 请先创建虚拟环境并安装 requirements.txt。
    pause
    exit /b 1
)

echo FastAPI 服务启动后可访问 http://127.0.0.1:8000/docs
".venv\Scripts\python.exe" -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
pause
