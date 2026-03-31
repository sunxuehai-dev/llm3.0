import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routers import (
    chat, models, images, audio, video, file_upload, datasets,
    auth, users, resources, user_files, reports, knowledge_base, monitor,
    competitions, exams, llmfactory, code_online, rag_retrieval
)

load_dotenv()

# 创建 FastAPI 应用
app = FastAPI(
    title="Universal Model Playground Gateway & User Management System",
    description="Combined API for AI models and user management",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """初始化数据库表"""
    from sqlalchemy import text
    from convert_url import Base as ConvertBase
    from app.database import convert_engine, UserBase, user_engine

    # 初始化 Convert URL 数据库
    ConvertBase.metadata.create_all(bind=convert_engine)

    # 为已有 training_jobs 表补充 dataset_id、data_type 列（若不存在）
    for col_name in ("dataset_id", "data_type"):
        try:
            with convert_engine.connect() as conn:
                conn.execute(text(f"ALTER TABLE training_jobs ADD COLUMN {col_name} VARCHAR(64)"))
                conn.commit()
        except Exception:
            pass

    # 确保用户库 ORM（含 rag_vector_kb_control 等）已注册到 metadata
    import app.models.user as _user_models_for_metadata  # noqa: F401

    # 初始化用户管理数据库
    UserBase.metadata.create_all(bind=user_engine)

    # LangGraph RAG（可选，失败时记录警告，不影响主服务）
    try:
        from app.services.rag import initialize_rag

        initialize_rag()
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("RAG 启动初始化异常（已忽略）: %s", e)


# 注册路由 - AI 模型服务
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(images.router)
app.include_router(audio.router)
app.include_router(video.router)
app.include_router(file_upload.router)
app.include_router(datasets.router)
app.include_router(llmfactory.router)
app.include_router(code_online.router)

# 注册路由 - 用户管理服务（使用 /api/v1 前缀）
app.include_router(auth.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(resources.router, prefix="/api/v1")
app.include_router(user_files.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(knowledge_base.router, prefix="/api/v1")
app.include_router(rag_retrieval.router, prefix="/api/v1")
app.include_router(monitor.router, prefix="/api/v1")
app.include_router(competitions.router, prefix="/api/v1")
app.include_router(exams.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Welcome to Universal Model Playground Gateway & User Management System",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Model Playground Gateway & User Management System"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
