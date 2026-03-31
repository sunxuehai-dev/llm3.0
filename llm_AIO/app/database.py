"""
统一数据库配置
支持多个数据库连接
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from app.config import get_settings

settings = get_settings()

# 用户管理数据库（LLM项目）
USER_DATABASE_URL = settings.DATABASE_URL
user_engine = create_engine(
    USER_DATABASE_URL,
    connect_args={"check_same_thread": False} if 'sqlite' in USER_DATABASE_URL else {}
)
UserSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=user_engine)
UserBase = declarative_base()

# Convert URL 数据库（Connect项目）
CONVERT_DATABASE_URL = settings.CONVERT_URL_DATABASE_URL
convert_engine = create_engine(
    CONVERT_DATABASE_URL,
    connect_args={"check_same_thread": False} if 'sqlite' in CONVERT_DATABASE_URL else {}
)
ConvertSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=convert_engine)

# 向后兼容：默认使用用户数据库
engine = user_engine
SessionLocal = UserSessionLocal
Base = UserBase


def get_db():
    """获取用户管理数据库会话"""
    db = UserSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_convert_db():
    """获取 Convert URL 数据库会话"""
    db = ConvertSessionLocal()
    try:
        yield db
    finally:
        db.close()
