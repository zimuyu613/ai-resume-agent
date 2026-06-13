@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo AI Resume Agent 启动脚本
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv\Scripts\python.exe
    echo 请先创建虚拟环境并安装依赖：
    echo python -m venv .venv
    echo .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo 正在启动 Streamlit 应用...
echo 如果浏览器没有自动打开，请访问终端中显示的 Local URL。
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

pause
