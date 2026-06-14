@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo   量化交易监控系统 — 一键启动
echo ========================================
echo.

:: 检查 Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

:: 检查 Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Node.js，请先安装 Node 18+
    pause
    exit /b 1
)

:: 启动后端
echo [1/2] 启动后端服务 (http://localhost:8000) ...
start "后端服务" cmd /k "cd /d %~dp0web\backend && python -m uvicorn app:app --reload --port 8000"

:: 等待后端初始化
echo       等待后端初始化...
timeout /t 3 /nobreak >nul

:: 启动前端
echo [2/2] 启动前端服务 (http://localhost:5173) ...
start "前端服务" cmd /k "cd /d %~dp0web\frontend && npm run dev"

echo.
echo ========================================
echo   启动完成！
echo   后端 API:  http://localhost:8000
echo   前端页面:  http://localhost:5173
echo   API 文档:  http://localhost:8000/docs
echo ========================================
echo.
echo 关闭此窗口不会影响服务，关闭对应的 cmd 窗口即可停止服务
pause
