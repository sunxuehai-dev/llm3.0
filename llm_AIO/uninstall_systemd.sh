#!/bin/bash
# =========================================
# Connect 项目 - Systemd 服务卸载脚本
# 统一网关服务卸载
# 自动停止、禁用并删除 systemd 服务
# =========================================

# 检查是否使用 bash 执行
if [ -z "$BASH_VERSION" ]; then
    echo "错误: 此脚本必须使用 bash 执行"
    echo "请使用: bash uninstall_systemd.sh"
    echo "或者: sudo bash uninstall_systemd.sh"
    exit 1
fi

set -euo pipefail

echo "====================================="
echo "  Unified Gateway Systemd 服务卸载"
echo "  (AI模型 + 用户管理系统)"
echo "====================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SERVICE_NAME="unified-gateway"
SERVICE_FILE="/etc/systemd/system/$SERVICE_NAME.service"

# 检查是否为 root 用户
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}[ERROR] 请使用 sudo 运行此脚本${NC}"
    echo "用法: sudo bash uninstall_systemd.sh"
    exit 1
fi

# 检查服务是否存在
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${YELLOW}[WARN] 服务文件不存在: $SERVICE_FILE${NC}"
    echo "[INFO] 服务可能已经卸载"
    exit 0
fi

# 检查服务状态
echo "[INFO] 检查服务状态..."
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo -e "${YELLOW}[INFO] 服务正在运行，正在停止...${NC}"
    systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}[INFO] 服务已停止${NC}"
else
    echo "[INFO] 服务未运行"
fi

# 禁用开机自启
echo "[INFO] 检查开机自启状态..."
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    echo -e "${YELLOW}[INFO] 服务已启用开机自启，正在禁用...${NC}"
    systemctl disable "$SERVICE_NAME"
    echo -e "${GREEN}[INFO] 开机自启已禁用${NC}"
else
    echo "[INFO] 服务未启用开机自启"
fi

# 删除服务文件
echo "[INFO] 删除服务文件..."
if [ -f "$SERVICE_FILE" ]; then
    rm -f "$SERVICE_FILE"
    echo -e "${GREEN}[INFO] 服务文件已删除: $SERVICE_FILE${NC}"
else
    echo "[INFO] 服务文件不存在"
fi

# 重新加载 systemd
echo "[INFO] 重新加载 systemd 配置..."
systemctl daemon-reload
systemctl reset-failed 2>/dev/null || true
echo -e "${GREEN}[INFO] systemd 配置已重新加载${NC}"

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  卸载完成！${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "服务已成功卸载："
echo "  - 服务已停止"
echo "  - 开机自启已禁用"
echo "  - 服务文件已删除"
echo ""
echo "注意："
echo "  - 项目文件和数据不会被删除"
echo "  - .env 配置文件仍然保留"
echo "  - 数据库文件仍然保留"
echo "  - 日志文件仍然保留在 logs/ 目录中"
echo "  - 如果需要，可以手动删除项目目录"
echo ""
