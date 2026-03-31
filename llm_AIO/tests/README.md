# 测试说明

本目录包含统一网关服务的所有测试脚本。

## 📋 测试文件列表

### AI 模型服务测试

| 测试文件 | 说明 |
|---------|------|
| `test_text_models_comprehensive.py` | 文本模型测试（qwen-max, qwen-plus, qwen-turbo） |
| `test_vision_models_comprehensive.py` | 视觉理解模型测试（qwen-vl-max, qwen-vl-plus, qwen3-vl-plus, qwen3-vl-flash） |
| `test_image_gen_models_comprehensive.py` | 图像生成模型测试（qwen-image-max） |
| `test_video_gen_models_comprehensive.py` | 视频生成模型测试（wan2.6-i2v-flash, wan2.6-i2v） |
| `test_omni_models_comprehensive.py` | 全模态模型测试（qwen3-omni-flash） |
| `test_file_upload.py` | 文件上传功能测试（视频/音频/图片） |

### 用户管理服务测试

| 测试文件 | 说明 |
|---------|------|
| `test_user_management.py` | 用户管理服务测试（认证、用户管理、资源管理、监控等） |

## 🚀 运行测试

### 前置条件

1. **启动服务**
   ```bash
   # 在项目根目录
   ./start.sh
   
   # 或
   python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **初始化数据库**（首次运行）
   ```bash
   # 在项目根目录
   python3 init_db.py
   ```

3. **配置环境变量**
   - 确保 `.env` 文件中配置了必要的 API Key
   - 用户管理服务需要默认管理员账户（`admin` / `admin123`）

### 运行所有测试

```bash
# 在 tests 目录下
chmod +x run_tests.sh
./run_tests.sh
```

### 运行单个测试

```bash
# AI 模型服务测试
python3 test_text_models_comprehensive.py
python3 test_vision_models_comprehensive.py
python3 test_image_gen_models_comprehensive.py
python3 test_video_gen_models_comprehensive.py
python3 test_omni_models_comprehensive.py
python3 test_file_upload.py

# 用户管理服务测试
python3 test_user_management.py
```

## 📝 测试说明

### AI 模型服务测试

- **端口**: `8000`
- **基础 URL**: `http://localhost:8000/api/playground`
- **认证**: 无需认证
- **测试内容**:
  - 模型列表获取
  - 文本对话（流式输出）
  - 视觉理解
  - 图像生成
  - 视频生成
  - 全模态对话
  - 文件上传（视频/音频/图片）

### 用户管理服务测试

- **端口**: `8000`
- **基础 URL**: `http://localhost:8000/api/v1`
- **认证**: 需要 JWT Token（使用默认管理员账户登录）
- **测试内容**:
  - 用户登录
  - 获取当前用户信息
  - 用户列表查询
  - 资源管理
  - 系统监控
  - 知识库查询

## ⚠️ 注意事项

1. **服务必须运行**: 所有测试都是针对运行中的服务进行的，确保服务已启动
2. **API Key 配置**: AI 模型服务测试需要配置相应的 API Key（`.env` 文件）
3. **数据库初始化**: 用户管理服务测试需要先运行 `init_db.py` 初始化数据库
4. **测试数据**: 测试可能会创建一些临时数据，不会影响生产环境
5. **网络连接**: 部分测试需要访问外部 API（如阿里云 DashScope），确保网络连接正常

## 🔍 测试结果

测试结果会输出到控制台，包括：
- ✅ PASS: 测试通过
- ❌ FAIL: 测试失败（会显示错误信息）

部分测试文件会在 `test_results/` 目录下生成测试结果文件。

## 📞 问题排查

### 服务连接失败

```
❌ 无法连接到服务器
```

**解决方案**:
1. 检查服务是否启动: `curl http://localhost:8000/health`
2. 检查端口是否正确（应为 8000）
3. 检查防火墙设置

### 认证失败

```
❌ 登录失败
```

**解决方案**:
1. 运行 `python3 init_db.py` 初始化数据库
2. 检查默认管理员账户是否存在
3. 检查密码是否正确（默认: `admin123`）

### API Key 错误

```
❌ API 调用失败
```

**解决方案**:
1. 检查 `.env` 文件中的 API Key 配置
2. 确认 API Key 有效且有足够的配额
3. 检查网络连接

---

**最后更新**: 2026-01-28
