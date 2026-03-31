from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum, JSON
from sqlalchemy.orm import relationship
import enum
from typing import Optional

from app.database import UserBase as Base
from app.datetime_utils import utc_now


class UserRole(str, enum.Enum):
    SUPER_ADMIN = "super_admin"
    TEACHER = "teacher"
    STUDENT = "student"


class ReportStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    GRADED = "graded"
    RETURNED = "returned"


class FileType(str, enum.Enum):
    MATERIAL = "material"
    ASSIGNMENT = "assignment"
    SUBMISSION = "submission"
    KNOWLEDGE_BASE = "knowledge_base"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    student_id = Column(String(50), unique=True, index=True, nullable=True)
    teacher_id = Column(String(50), unique=True, index=True, nullable=True)
    
    assigned_resources = relationship("ResourceAllocation", back_populates="user")
    sent_files = relationship("File", foreign_keys="File.sender_id", back_populates="sender")
    received_files = relationship("File", foreign_keys="File.receiver_id", back_populates="receiver")
    submitted_reports = relationship("Report", foreign_keys="Report.student_id", back_populates="student")
    graded_reports = relationship("Report", foreign_keys="Report.teacher_id", back_populates="teacher")
    knowledge_base_entries = relationship("KnowledgeBaseEntry", back_populates="uploader")


class SystemResource(Base):
    __tablename__ = "system_resources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    resource_type = Column(String(50))
    total_amount = Column(Float)
    unit = Column(String(20))
    created_at = Column(DateTime, default=utc_now)


class ResourceAllocation(Base):
    __tablename__ = "resource_allocations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    resource_id = Column(Integer, ForeignKey("system_resources.id"), nullable=False)
    allocated_amount = Column(Float, nullable=False)
    allocated_at = Column(DateTime, default=utc_now)
    expires_at = Column(DateTime, nullable=True)
    notes = Column(Text)
    
    user = relationship("User", back_populates="assigned_resources")
    resource = relationship("SystemResource")


class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    receiver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_batch = Column(Boolean, default=False)
    batch_group_id = Column(String(50), nullable=True)
    description = Column(Text)
    created_at = Column(DateTime, default=utc_now)
    
    sender = relationship("User", foreign_keys=[sender_id], back_populates="sent_files")
    receiver = relationship("User", foreign_keys=[receiver_id], back_populates="received_files")


class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    status = Column(String(20), default="submitted")
    grade = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    submitted_at = Column(DateTime, default=utc_now)
    graded_at = Column(DateTime, nullable=True)
    
    student = relationship("User", foreign_keys=[student_id], back_populates="submitted_reports")
    teacher = relationship("User", foreign_keys=[teacher_id], back_populates="graded_reports")
    file = relationship("File")


class KnowledgeBaseEntry(Base):
    __tablename__ = "knowledge_base_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    category = Column(String(100))
    tags = Column(JSON)
    uploader_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_indexed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    uploader = relationship("User", back_populates="knowledge_base_entries")
    file = relationship("File")


class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float)
    unit = Column(String(20))
    recorded_at = Column(DateTime, default=utc_now)
    extra_data = Column(JSON, nullable=True)


class RagVectorKbControl(Base):
    """RAG 向量知识库运行时开关（单例行 id=1）。与进程级 RAG_ENABLED 独立：用于教师/管理员暂停入库与对话。"""

    __tablename__ = "rag_vector_kb_control"

    id = Column(Integer, primary_key=True)
    vector_kb_disabled = Column(Boolean, nullable=False, default=False)
    note = Column(Text, nullable=True)
    updated_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
