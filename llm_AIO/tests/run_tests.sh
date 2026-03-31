#!/bin/bash
# =========================================
# 运行所有测试 (服务器端验证)
# 请确保服务器已启动: ./start.sh
# 端口: 8000
# =========================================

echo "====================================="
echo "  Universal Gateway API - 完整测试"
echo "====================================="
echo ""

# --- 切换到脚本所在目录 ---
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

# 检查服务是否运行
echo "检查服务状态..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ 错误: 服务未启动！"
    echo "请先启动服务: ./start.sh 或 python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi
echo "✅ 服务运行正常"
echo ""

echo "====================================="
echo "  AI 模型服务测试"
echo "====================================="
echo ""

echo "[1/6] Testing Text Models..."
$PYTHON test_text_models_comprehensive.py
echo ""

echo "[2/6] Testing Vision Models..."
$PYTHON test_vision_models_comprehensive.py
echo ""

echo "[3/6] Testing Image Gen Models..."
$PYTHON test_image_gen_models_comprehensive.py
echo ""

echo "[4/6] Testing Video Gen Models..."
$PYTHON test_video_gen_models_comprehensive.py
echo ""

echo "[5/6] Testing Omni Models..."
$PYTHON test_omni_models_comprehensive.py
echo ""

echo "[6/6] Testing File Upload..."
$PYTHON test_file_upload.py
echo ""

echo "====================================="
echo "  用户管理服务测试"
echo "====================================="
echo ""

if [ -f "test_user_management.py" ]; then
    echo "[1/1] Testing User Management..."
    $PYTHON test_user_management.py
    echo ""
else
    echo "⚠️  用户管理测试文件不存在，跳过"
    echo ""
fi

echo "====================================="
echo "  All Tests Completed!"
echo "====================================="
