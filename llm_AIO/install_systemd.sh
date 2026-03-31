#!/bin/bash
# =========================================
# Connect 项目 - Systemd 服务安装脚本
# 统一网关服务（AI模型 + 用户管理）
# 自动配置并安装 systemd 服务
# =========================================

# 检查是否使用 bash 执行
if [ -z "$BASH_VERSION" ]; then
    echo "错误: 此脚本必须使用 bash 执行"
    echo "请使用: bash install_systemd.sh"
    echo "或者: sudo bash install_systemd.sh"
    exit 1
fi

set -euo pipefail

echo "====================================="
echo "  Unified Gateway Systemd 服务安装"
echo "  (AI模型 + 用户管理系统)"
echo "====================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SERVICE_NAME="unified-gateway"
SERVICE_FILE="$PROJECT_DIR/$SERVICE_NAME.service"

echo "[INFO] 项目目录: $PROJECT_DIR"
echo "[INFO] 服务名称: $SERVICE_NAME"
echo ""

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}[ERROR] 请使用 sudo 运行此脚本${NC}"
    echo "用法: sudo bash install_systemd.sh"
    exit 1
fi

# 检查项目文件是否存在
if [ ! -f "$PROJECT_DIR/app/main.py" ]; then
    echo -e "${RED}[ERROR] 未找到 app/main.py，请确保在项目根目录运行此脚本${NC}"
    exit 1
fi

# 检测当前用户（如果不是 root，使用当前用户）
if [ -z "$SUDO_USER" ]; then
    RUN_USER="root"
else
    RUN_USER="$SUDO_USER"
fi

echo "[INFO] 检测到运行用户: $RUN_USER"

# 检测 Python 路径
echo "[INFO] 正在检测 Python 环境..."

# 尝试常见的 Python 路径
PYTHON_PATHS=(
    "/home/$RUN_USER/miniforge3/envs/llm/bin/python"
    "/home/$RUN_USER/anaconda3/envs/llm/bin/python"
    "/home/$RUN_USER/.conda/envs/llm/bin/python"
    "/usr/bin/python3"
    "/usr/local/bin/python3"
    "$(which python3)"
)

PYTHON_PATH=""
for path in "${PYTHON_PATHS[@]}"; do
    if [ -f "$path" ] && "$path" --version &> /dev/null; then
        PYTHON_PATH="$path"
        echo -e "${GREEN}[INFO] 找到 Python: $PYTHON_PATH${NC}"
        echo "[INFO] Python 版本: $($PYTHON_PATH --version)"
        break
    fi
done

if [ -z "$PYTHON_PATH" ]; then
    echo -e "${YELLOW}[WARN] 未自动检测到 Python，请手动输入 Python 路径:${NC}"
    read -p "Python 路径: " PYTHON_PATH
    if [ ! -f "$PYTHON_PATH" ]; then
        echo -e "${RED}[ERROR] Python 路径无效: $PYTHON_PATH${NC}"
        exit 1
    fi
fi

# 检测 Python 环境目录
PYTHON_ENV_DIR=$(dirname "$PYTHON_PATH")
PYTHON_BIN_DIR="$PYTHON_ENV_DIR"

# 检查依赖
echo "[INFO] 检查依赖..."
if ! "$PYTHON_PATH" -c "import uvicorn" &> /dev/null; then
    echo -e "${YELLOW}[WARN] uvicorn 未安装，正在安装依赖...${NC}"
    "$PYTHON_PATH" -m pip install -r "$PROJECT_DIR/requirements.txt"
fi

# 检查 OSS 存储依赖（如果使用 OSS）
if [ -f "$PROJECT_DIR/.env" ]; then
    if grep -q "STORAGE_TYPE=oss" "$PROJECT_DIR/.env" 2>/dev/null; then
        if ! "$PYTHON_PATH" -c "import oss2" &> /dev/null; then
            echo -e "${YELLOW}[WARN] 检测到使用 OSS 存储，但 oss2 未安装${NC}"
            echo "[INFO] 正在安装 oss2..."
            "$PYTHON_PATH" -m pip install oss2
        fi
    fi
fi

# 检查 .env 文件
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo -e "${YELLOW}[WARN] .env 文件不存在${NC}"
    if [ -f "$PROJECT_DIR/.env.example" ]; then
        echo "[INFO] 从 .env.example 创建 .env 文件..."
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        echo -e "${YELLOW}[WARN] 请编辑 .env 文件并配置 API Key 等信息${NC}"
        echo "   文件位置: $PROJECT_DIR/.env"
    else
        echo -e "${YELLOW}[WARN] .env.example 文件也不存在，请手动创建 .env 文件${NC}"
    fi
fi

# 创建日志目录
LOG_DIR="$PROJECT_DIR/logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "[INFO] 创建日志目录: $LOG_DIR"
    mkdir -p "$LOG_DIR"
    chown "$RUN_USER:$RUN_USER" "$LOG_DIR"
fi

# 检测端口（默认 8000）
read -p "请输入服务端口 [默认: 8000]: " PORT
PORT=${PORT:-8000}

# 检查端口是否被占用
if command -v netstat &> /dev/null; then
    if netstat -tuln | grep -q ":$PORT "; then
        echo -e "${YELLOW}[WARN] 端口 $PORT 已被占用${NC}"
        read -p "是否继续? (y/n): " CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            exit 1
        fi
    fi
fi

# 检查数据库初始化
echo "[INFO] 检查数据库初始化..."
if [ -f "$PROJECT_DIR/init_db.py" ]; then
    echo "[INFO] 发现 init_db.py，建议首次运行前执行数据库初始化"
    read -p "是否现在初始化数据库? (y/n) [默认: n]: " INIT_DB
    INIT_DB=${INIT_DB:-n}
    if [ "$INIT_DB" = "y" ]; then
        echo "[INFO] 正在初始化数据库..."
        "$PYTHON_PATH" "$PROJECT_DIR/init_db.py"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[INFO] 数据库初始化成功！${NC}"
        else
            echo -e "${YELLOW}[WARN] 数据库初始化失败，但继续安装服务${NC}"
        fi
    fi
fi

# 创建 systemd 服务文件内容
echo "[INFO] 生成 systemd 服务配置..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Unified Gateway Service - AI Models & User Management System
After=network.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PYTHON_BIN_DIR:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PYTHON_PATH -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/server.log
StandardError=append:$LOG_DIR/error.log
SyslogIdentifier=$SERVICE_NAME

# 安全设置
NoNewPrivileges=true
PrivateTmp=true

# 环境变量（如果需要，取消注释并修改）
# Environment="DASHSCOPE_API_KEY=your-api-key"
# Environment="STORAGE_TYPE=local"
# Environment="SERVER_URL=http://localhost:$PORT"
# Environment="SECRET_KEY=your-secret-key"
# Environment="DATABASE_URL=sqlite:///./user_management.db"

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}[INFO] 服务配置文件已生成: $SERVICE_FILE${NC}"
echo ""

# 复制服务文件到 systemd 目录
echo "[INFO] 复制服务文件到 /etc/systemd/system/..."
cp "$SERVICE_FILE" "/etc/systemd/system/$SERVICE_NAME.service"
chmod 644 "/etc/systemd/system/$SERVICE_NAME.service"

# 确保项目目录权限正确
echo "[INFO] 设置项目目录权限..."
chown -R "$RUN_USER:$RUN_USER" "$PROJECT_DIR"

# 重新加载 systemd
echo "[INFO] 重新加载 systemd 配置..."
systemctl daemon-reload

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "下一步操作："
echo ""
echo "1. 启动服务:"
echo "   sudo systemctl start $SERVICE_NAME"
echo ""
echo "2. 设置开机自启:"
echo "   sudo systemctl enable $SERVICE_NAME"
echo ""
echo "3. 查看服务状态:"
echo "   sudo systemctl status $SERVICE_NAME"
echo ""
echo "4. 查看服务日志:"
echo "   sudo journalctl -u $SERVICE_NAME -f"
echo "   或查看文件日志: tail -f $LOG_DIR/server.log"
echo ""
echo "5. 测试服务:"
echo "   curl http://localhost:$PORT/health"
echo "   curl http://localhost:$PORT/"
echo ""
echo "6. API 文档:"
echo "   http://localhost:$PORT/docs"
echo ""
echo "服务配置信息:"
echo "  - 服务名称: $SERVICE_NAME"
echo "  - 运行用户: $RUN_USER"
echo "  - 项目目录: $PROJECT_DIR"
echo "  - Python 路径: $PYTHON_PATH"
echo "  - 服务端口: $PORT"
echo "  - 服务文件: /etc/systemd/system/$SERVICE_NAME.service"
echo "  - 日志目录: $LOG_DIR"
echo ""
echo "功能说明:"
echo "  - AI 模型服务: /api/playground/*"
echo "  - 用户管理服务: /api/v1/*"
echo "  - 健康检查: /health"
echo ""
