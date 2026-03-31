#!/bin/bash
# =========================================
# Unified Gateway - 一键启动脚本
# AI模型服务 + 用户管理系统
# 适用于: Linux 服务器
# =========================================

echo "====================================="
echo "  Unified Gateway - Starting"
echo "  (AI Models + User Management)"
echo "====================================="

# --- 配置区域 (请根据服务器环境修改) ---
# PYTHON_PATH="${PYTHON_PATH:-python3}"
PYTHON_PATH="/home/llm/miniforge3/envs/llm/bin/python"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# --- 切换到脚本所在目录 ---
cd "$(dirname "$0")"

# --- 检查 Python ---
if ! command -v "$PYTHON_PATH" &> /dev/null; then
    echo "[ERROR] Python not found! Please install Python or set PYTHON_PATH."
    exit 1
fi
echo "[INFO] Python: $($PYTHON_PATH --version)"

# --- 检查项目文件 ---
if [ ! -f "app/main.py" ]; then
    echo "[ERROR] app/main.py not found! Please run this script from project root."
    exit 1
fi

# --- 检查 .env 文件 ---
if [ ! -f ".env" ]; then
    echo "[WARN] .env file not found! Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[INFO] Please edit .env and add your API keys:"
        echo "       nano .env"
    else
        echo "[WARN] .env.example not found! Using default configuration."
    fi
fi

# --- 检查依赖 ---
if ! $PYTHON_PATH -c "import uvicorn" &> /dev/null; then
    echo "[WARN] Installing dependencies..."
    $PYTHON_PATH -m pip install -r requirements.txt
fi

# --- 初始化数据库 (如果不存在) ---
if [ -f "init_db.py" ]; then
    # 检查用户管理数据库是否存在
    DB_FILE="${DATABASE_URL:-user_management.db}"
    if [[ "$DB_FILE" == sqlite* ]]; then
        # 从 DATABASE_URL 提取数据库文件名
        DB_NAME=$(echo "$DB_FILE" | sed 's|sqlite:///\./||' | sed 's|sqlite:///||')
    else
        DB_NAME="user_management.db"
    fi
    
    if [ ! -f "$DB_NAME" ]; then
        echo "[INFO] Database not found. Initializing database..."
        $PYTHON_PATH init_db.py
        if [ $? -eq 0 ]; then
            echo "[INFO] Database initialized successfully!"
            echo "[INFO] Default admin user created:"
            echo "       Username: admin"
            echo "       Password: admin123"
            echo "       ⚠️  Please change the password after first login!"
        else
            echo "[WARN] Database initialization failed, but continuing..."
        fi
    else
        echo "[INFO] Database already exists, skipping initialization."
    fi
fi

# --- 启动服务器 ---
echo ""
echo "[INFO] Starting server on http://$HOST:$PORT"
echo "[INFO] API Documentation: http://$HOST:$PORT/docs"
echo "[INFO] Health Check: http://$HOST:$PORT/health"
echo ""
echo "Available endpoints:"
echo "  - AI Models: http://$HOST:$PORT/api/playground/*"
echo "  - User Management: http://$HOST:$PORT/api/v1/*"
echo ""
echo "[INFO] Press Ctrl+C to stop."
echo ""

$PYTHON_PATH -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload