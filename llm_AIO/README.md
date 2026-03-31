# Universal Model Playground Gateway & User Management System

> 统一网关服务，整合 AI 模型服务和用户管理系统，为前端提供统一的 API 接口。

---

## 📖 阅读顺序 (前端必看)

| 顺序 | 文件 | 说明 |
|------|------|------|
| 1️⃣ | **[API_接口文档.md](API_接口文档.md)** | 完整接口文档（包含 AI 模型服务、用户管理服务与竞赛报名系统） |
| 2️⃣ | [.env.example](.env.example) | 环境变量模板（复制为 .env 后按需修改） |
| 3️⃣ | [available_models_cn.md](available_models_cn.md) | 可用模型列表说明 |
| 4️⃣ | [LOCAL_MODELS_DEPLOY.md](LOCAL_MODELS_DEPLOY.md) | 本地模型（如 Ollama）部署说明 |
| 5️⃣ | [tests/README.md](tests/README.md) | 测试说明与运行方式 |

---

## 🕐 时间与 UTC 约定

- 服务端存储与业务比较统一使用 **UTC**。
- **JSON 响应里的时间字符串**由环境变量 **`API_RESPONSE_DATETIME_TZ`** 控制（见 `.env.example`）：
  - **默认 `Asia/Shanghai`**：输出带 **`+08:00`** 的 ISO8601，墙钟时间与北京时间一致（适合前端不做时区转换时直接显示）。
  - 设为 **`UTC`**：输出带 **`Z`** 的 UTC 时间（适合国际化或前端自行 `toLocal()`）。
- 请求体里的 `start_at`、`end_at` 等建议使用 **带时区的 ISO8601**（`Z` 或 `+08:00`），避免歧义。

实现说明见 `app/datetime_utils.py`。

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.12
- 数据库：SQLite（默认，无需额外配置）或 PostgreSQL/MySQL（可选）

### 2. 安装依赖

```bash
# 安装 Python 依赖
pip3 install -r requirements.txt
```

### 3. 配置环境变量

```bash
# Linux/Mac
cp .env.example .env

# Windows
copy .env.example .env

# 编辑 .env 文件，填入必要的 API Key 和密钥
```

**主要配置**（详见 [.env.example](.env.example)）：
- **AI 模型**：`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`、`VOLCENGINE_API_KEY`
- **语音**：`XUNFEI_APP_ID`、`XUNFEI_SECRET_KEY`（讯飞语音转写，可选）
- **用户系统**：`SECRET_KEY`、`DATABASE_URL`、`ACCESS_TOKEN_EXPIRE_MINUTES`
- **文件存储**：`STORAGE_TYPE`（local/oss）、`SERVER_URL`（大模型需访问文件时须填公网地址）、OSS 相关（选填）
- **双数据库**：`DATABASE_URL`（用户库）、`CONVERT_URL_DATABASE_URL`（文件元数据库）

### 4. 初始化数据库

```bash
# 初始化用户管理数据库（创建表、默认管理员、系统资源）
python3 init_db.py
```

**默认管理员账户**：
- 用户名：`admin`
- 密码：`admin123`
- ⚠️ **首次登录后请立即修改密码！**

### 5. 启动服务

**方式一：使用启动脚本（推荐）**

```bash
# Linux/Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

**方式二：使用 systemd 服务（生产环境推荐）**

```bash
# 安装 systemd 服务
chmod +x install_systemd.sh
sudo ./install_systemd.sh

# 启动服务
sudo systemctl start unified-gateway

# 查看服务状态
sudo systemctl status unified-gateway

# 查看日志
sudo journalctl -u unified-gateway -f
```

**方式三：命令行启动**

```bash
# 开发环境
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 返回示例
# {"status": "ok", "service": "Model Playground Gateway & User Management System"}

# 访问 API 文档
# 浏览器打开: http://localhost:8000/docs
```

---

## 📁 项目结构

```
llm_AIO/
├── app/                         # 后端核心代码
│   ├── main.py                  # FastAPI 应用入口
│   ├── config.py                # 配置管理
│   ├── database.py              # 数据库配置（双数据库支持）
│   ├── security.py              # 认证和安全
│   ├── schemas.py               # Pydantic 模型
│   ├── permissions.py           # 权限管理
│   ├── adapters/                # AI 模型服务提供商适配器
│   │   ├── aliyun.py            # 阿里云 DashScope（文本/多模态）
│   │   ├── aliyun_audio.py      # 阿里云语音（TTS/ASR）
│   │   ├── aliyun_image.py      # 阿里云图像生成
│   │   ├── aliyun_video.py      # 阿里云视频生成
│   │   ├── deepseek.py          # DeepSeek
│   │   ├── doubao.py            # 豆包
│   │   ├── xunfei_audio.py      # 讯飞语音转写
│   │   └── local.py             # 本地模型（如 Ollama）
│   ├── models/                  # 数据模型
│   │   ├── user.py              # 用户管理模型
│   │   ├── competition.py      # 竞赛报名系统模型
│   │   ├── chat.py              # 对话模型
│   │   └── multimodal.py        # 多模态模型
│   ├── routers/                 # API 路由
│   │   ├── chat.py              # 对话接口
│   │   ├── models.py            # 模型列表
│   │   ├── images.py            # 图像生成、提示词模板
│   │   ├── audio.py             # 语音合成与语音识别
│   │   ├── video.py             # 视频生成
│   │   ├── file_upload.py       # 文件上传（video/audio/image）
│   │   ├── llmfactory.py        # LlamaFactory 微调/合并/推理 API（可选）
│   │   ├── auth.py              # 认证接口
│   │   ├── users.py             # 用户管理
│   │   ├── resources.py        # 资源管理
│   │   ├── user_files.py       # 用户文件（个人/批量发送）
│   │   ├── reports.py           # 报告管理
│   │   ├── knowledge_base.py    # 知识库
│   │   └── monitor.py           # 系统监控
│   │   └── competitions.py     # 竞赛报名/组队/提交/评分
│   └── services/                # 业务逻辑
│       ├── model_factory.py    # 模型工厂
│       ├── registry.py         # 模型注册表
│       ├── file_processing.py  # Base64 转 URL 等文件处理
│       └── llmfactory_service.py # LlamaFactory 训练/合并/推理封装
├── convert_url/                 # 文件 URL 转换工具
│   ├── __init__.py
│   └── core.py
├── tests/                       # 测试脚本（见 tests/README.md）
├── test_files/                  # 测试用示例文件
├── init_db.py                   # 数据库初始化脚本
├── .env.example                 # 环境变量模板（复制为 .env）
├── requirements.txt             # Python 依赖
├── start.sh / start.bat         # 启动脚本（Linux/Mac / Windows）
├── install_systemd.sh           # systemd 服务安装（Linux）
├── uninstall_systemd.sh         # systemd 服务卸载
├── available_models_cn.md       # 可用模型说明
├── LOCAL_MODELS_DEPLOY.md       # 本地模型部署说明
├── API_接口文档.md              # 📌 完整 API 文档
└── README.md                    # 项目说明（本文件）
```

---

## 🔗 API 端点概览

### AI 模型服务（无需认证）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/playground/models` | GET | 获取模型列表 |
| `/api/playground/chat` | POST | 文本/视觉/全模态对话（SSE 流式，支持 Base64 转 URL） |
| `/api/playground/images/generations` | POST | 图像生成 |
| `/api/playground/images/prompts/templates` | GET | 教育场景提示词模板列表 |
| `/api/playground/videos/generations` | POST | 视频生成（支持 Base64 图转 URL） |
| `/api/playground/audio/speech` | POST | 文本转语音 |
| `/api/playground/audio/transcription` | POST | 语音识别（ASR，支持 URL/服务器路径，Base64 自动转 URL） |
| `/api/{file_category}/upload` | POST | 文件上传（file_category: video/audio/image） |
| `/api/{file_category}/list` | GET | 获取文件列表 |
| `/api/{file_category}/{file_id}` | GET | 获取文件 URL |
| `/api/{file_category}/{file_id}` | DELETE | 删除文件 |
| `/api/playground/llmfactory/*` | POST | LlamaFactory 微调/合并/推理（可选，见 API 文档） |

### 用户管理服务（需要 JWT Token 认证）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/login` | POST | 用户登录 |
| `/api/v1/auth/me` | GET | 获取当前用户信息 |
| `/api/v1/auth/refresh-token` | POST | 刷新令牌 |
| `/api/v1/users/` | GET | 获取用户列表 |
| `/api/v1/users/{user_id}` | GET | 获取用户详情 |
| `/api/v1/users/` | POST | 创建用户 |
| `/api/v1/users/{user_id}` | PUT | 更新用户 |
| `/api/v1/users/{user_id}` | DELETE | 删除用户 |
| `/api/v1/resources/` | GET | 获取资源列表 |
| `/api/v1/resources/allocate` | POST | 分配资源 |
| `/api/v1/user-files/upload` | POST | 上传文件 |
| `/api/v1/user-files/send` | POST | 发送文件（支持个人/批量） |
| `/api/v1/reports/` | POST | 提交报告 |
| `/api/v1/reports/{report_id}/grade` | PUT | 评分报告 |
| `/api/v1/knowledge-base/` | POST | 创建知识库条目 |
| `/api/v1/knowledge-base/` | GET | 获取知识库列表 |
| `/api/v1/monitor/health` | GET | 系统健康状态 |
| `/api/v1/monitor/gpu` | GET | GPU 监控 |
| `/api/v1/competitions/` | POST | 创建竞赛（管理员） |
| `/api/v1/competitions/{competition_id}/publish` | PUT | 发布竞赛（管理员） |
| `/api/v1/competitions/enroll` | POST | 报名参赛（学生） |
| `/api/v1/competitions/teams` | POST | 创建队伍（学生，成为队长） |
| `/api/v1/competitions/teams/{team_id}/members` | POST | 加入队伍（学生，队员） |
| `/api/v1/competitions/teams/{team_id}/transfer-captain` | POST | 队长转让（学生） |
| `/api/v1/competitions/teams/{team_id}/leave` | POST | 队长退队（先转让） |
| `/api/v1/competitions/submissions` | POST | 提交作品（学生：个人/队伍） |
| `/api/v1/competitions/submissions/{submission_id}/review-grade` | PUT | 评分/审核（教师/评委） |

**完整 API 文档**：请查看 [API_接口文档.md](API_接口文档.md)

---

## 🔐 认证说明

### AI 模型服务

- **无需认证**：所有 `/api/playground/*` 接口无需认证
- API 密钥在后端配置（`.env` 文件）

### 用户管理服务

- **需要 JWT Token 认证**：所有 `/api/v1/*` 接口需要认证
- **获取 Token**：
  ```bash
  curl -X POST "http://localhost:8000/api/v1/auth/login" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=admin&password=admin123"
  ```
- **使用 Token**：
  ```bash
  curl -X GET "http://localhost:8000/api/v1/auth/me" \
    -H "Authorization: Bearer <access_token>"
  ```

### 角色权限

| 角色 | 说明 | 权限范围 |
|------|------|----------|
| `super_admin` | 超级管理员 | 所有权限 |
| `teacher` | 教师 | 管理学生、批改报告、上传知识库等 |
| `student` | 学生 | 提交报告、接收文件、查看知识库 |

---

## 💻 前端调用示例

### AI 模型服务（无需认证）

```javascript
// 获取模型列表
const models = await fetch('http://localhost:8000/api/playground/models')
  .then(r => r.json());

// 文本对话（流式）
const response = await fetch('http://localhost:8000/api/playground/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    provider: 'aliyun',
    model: 'qwen-max',
    messages: [{ role: 'user', content: '你好' }],
    config: { temperature: 0.7, stream: true }
  })
});

// 处理 SSE 流式响应
const reader = response.body.getReader();
const decoder = new TextDecoder();
// ... 处理流式数据

// 上传文件
const formData = new FormData();
formData.append('file', file);
formData.append('user_id', 'user123');

const uploadResponse = await fetch('http://localhost:8000/api/video/upload', {
  method: 'POST',
  body: formData
});
const uploadData = await uploadResponse.json();
// 返回中含 video_id、url 等，file_category 为 audio/image 时对应 audio_id/image_id
console.log('文件URL:', uploadData.url);
```

### 用户管理服务（需要认证）

```javascript
// 登录获取 Token
const loginResponse = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    username: 'admin',
    password: 'admin123'
  })
});
const { access_token } = await loginResponse.json();

// 使用 Token 调用 API
const usersResponse = await fetch('http://localhost:8000/api/v1/users/', {
  headers: {
    'Authorization': `Bearer ${access_token}`
  }
});
const users = await usersResponse.json();

// 上传文件（需要认证）
const formData = new FormData();
formData.append('file', file);
formData.append('file_type', 'material');

const fileResponse = await fetch('http://localhost:8000/api/v1/user-files/upload', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`
  },
  body: formData
});
```

**更多示例**：请查看 [API_接口文档.md](API_接口文档.md)

---

## ⚠️ 重要：文件 URL 配置说明

### 为什么 localhost URL 不能用于大模型 API？

当使用 `STORAGE_TYPE=local` 时，生成的 URL 格式为：
```
http://localhost:8000/api/video/xxx
```

**问题**：
- `localhost` 只能在本机访问
- 大模型 API 服务运行在云端，**无法访问你本地的 localhost**
- 如果上传文件后需要将 URL 传给大模型 API，必须使用**公网可访问的地址**

### 解决方案

#### 方案 1：配置公网 SERVER_URL

在 `.env` 文件中配置 `SERVER_URL` 为公网可访问的地址：

```bash
# 选项 A: 使用公网 IP
SERVER_URL=http://123.456.789.0:8000

# 选项 B: 使用域名
SERVER_URL=http://your-domain.com

# 选项 C: 使用内网穿透工具 (ngrok/frp)
SERVER_URL=https://abc123.ngrok.io
```

**注意事项**：
- 确保服务器防火墙开放 8000 端口
- 确保服务启动时使用 `--host 0.0.0.0`（已在 `start.sh` 中配置）

#### 方案 2：使用 OSS 存储（推荐）

OSS 存储直接返回公网可访问的 URL，无需配置：

```bash
# 首先安装 oss2（阿里云官方 SDK）
pip install oss2

# 配置 .env 文件
STORAGE_TYPE=oss
OSS_ENDPOINT=https://oss-cn-beijing.aliyuncs.com
OSS_ACCESS_KEY_ID=your_key
OSS_ACCESS_KEY_SECRET=your_secret
OSS_BUCKET_NAME=your_bucket
OSS_REGION=oss-cn-beijing
```

**优势**：
- ✅ 自动生成公网可访问的 URL
- ✅ 无需配置 SERVER_URL
- ✅ 支持 CDN 加速
- ✅ 不占用服务器存储空间

#### 方案 3：使用内网穿透工具

**ngrok 示例**：
```bash
# 安装 ngrok
# 启动隧道
ngrok http 8000

# 将生成的 URL 配置到 .env
SERVER_URL=https://abc123.ngrok.io
```

### 验证配置

上传文件后，检查返回的 URL 是否为公网可访问：

```bash
# 测试 URL 是否可访问
curl http://your-public-url/api/video/xxx

# 如果返回文件内容，说明配置成功
```

---

## 🗄️ 数据库说明

### 双数据库架构

项目使用双数据库架构：

1. **用户管理数据库**（`user_management.db`）
   - 存储用户、资源、文件、报告、知识库等数据
   - 配置：`DATABASE_URL=sqlite:///./user_management.db`

2. **文件转换数据库**（`videos.db`）
   - 存储上传的视频、音频、图片文件元数据
   - 配置：`CONVERT_URL_DATABASE_URL=sqlite:///./videos.db`

### 初始化数据库

首次运行前，需要初始化数据库：

```bash
python3 init_db.py
```

该脚本会：
- 创建所有数据库表
- 创建默认超级管理员账户（`admin` / `admin123`）
- 初始化系统资源（CPU、GPU、存储）

### 数据库迁移

如需使用 PostgreSQL 或 MySQL，修改 `.env` 中的数据库 URL：

```env
# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/dbname

# MySQL
DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
```

---

## 🛠️ 开发指南

### 添加新的 AI 模型

1. 在 `app/adapters/` 中创建适配器
2. 在 `app/services/registry.py` 中注册模型
3. 更新 `API_接口文档.md`

### 添加新的用户管理功能

1. 在 `app/models/user.py` 中定义数据模型
2. 在 `app/routers/` 中创建路由
3. 在 `app/main.py` 中注册路由
4. 更新 `API_接口文档.md`

### 运行测试

```bash
cd tests
chmod +x run_tests.sh
./run_tests.sh
```

---

## 📦 部署

### 使用 systemd（推荐）

```bash
# 安装服务
sudo ./install_systemd.sh

# 启动服务
sudo systemctl start unified-gateway

# 设置开机自启
sudo systemctl enable unified-gateway

# 查看状态
sudo systemctl status unified-gateway

# 查看日志
sudo journalctl -u unified-gateway -f
```

### 使用 Docker（可选）

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
``

如有问题，请查看：
- **API 文档**：[API_接口文档.md](API_接口文档.md)
- **环境变量**：[.env.example](.env.example)
- **测试说明**：[tests/README.md](tests/README.md)
- **Swagger UI**：`http://localhost:8000/docs`

---

## 📝 更新日志

### v3.0.2 (2026-02-04)
- ✅ **文件处理**：新增 `file_processing` 服务，支持 Base64 自动转 URL（对话/语音/视频等）
- ✅ **语音/视频**：`audio/transcription`、`video/generations` 支持 Base64 输入自动上传转 URL
- ✅ **Bug 修复**：audio.py 补充 Depends/Session/file_processing 导入；video.py 修正导入位置与语法
- ✅ **README**：更新项目结构、API 概览、讯飞配置、提示词模板与 llmfactory 说明

### v3.0.1 (2026-01-30)
- ✅ **用户文件**：发送文件支持个人发送与批量发送（`receiver_id` / `batch_data_json`）
- ✅ **报告与知识库**：提交报告、创建知识库条目使用 Form 参数，支持 multipart 上传
- ✅ **环境变量**：新增 `.env.example` 模板，涵盖 AI 模型、用户系统、文件存储、双数据库
- ✅ **文档**：README 与 API 文档与当前项目结构对齐，补充语音识别、批量发送等说明

### v3.0.0 (2026-01-28)
- ✅ **项目合并**：整合 AI 模型服务和用户管理系统到统一网关
- ✅ **端口统一**：所有服务统一使用 8000 端口
- ✅ **路径规范**：AI 模型服务使用 `/api/playground/*`，用户管理服务使用 `/api/v1/*`
- ✅ **认证分离**：AI 模型服务无需认证，用户管理服务需要 JWT Token
- ✅ **文档完善**：包含所有接口的详细说明和示例代码
- ✅ **数据库初始化**：添加 `init_db.py` 脚本，自动创建默认管理员和系统资源

---

**文档版本**: v3.0.2  
**最后更新**: 2026-02-04  
**服务端口**: 8000
