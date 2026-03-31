from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Float, UniqueConstraint, JSON
from sqlalchemy.orm import relationship

from app.database import UserBase as Base
from app.datetime_utils import utc_now


class QuestionType:
    SINGLE = "single"       # 单选
    MULTIPLE = "multiple"   # 多选
    TRUE_FALSE = "true_false"  # 判断


class ExamStatus:
    DRAFT = "draft"
    PUBLISHED = "published"
    CLOSED = "closed"


class QuestionBankItem(Base):
    __tablename__ = "question_bank_items"

    id = Column(Integer, primary_key=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    question_type = Column(String(30), nullable=False)  # single/multiple/true_false
    stem = Column(Text, nullable=False)

    # 选项：JSON list，例如 [{"key":"A","text":"..."}, ...]
    options = Column(JSON, nullable=True)

    # 正确答案：
    # - single/true_false: "A" / "true"
    # - multiple: ["A","C"]
    correct_answer = Column(JSON, nullable=False)

    score = Column(Float, default=1.0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    creator = relationship("User", foreign_keys=[created_by])


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    status = Column(String(30), default=ExamStatus.DRAFT, nullable=False)

    start_at = Column(DateTime, nullable=True)
    end_at = Column(DateTime, nullable=True)
    duration_minutes = Column(Integer, default=60, nullable=False)

    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    creator = relationship("User", foreign_keys=[created_by])
    questions = relationship("ExamQuestion", back_populates="exam", cascade="all, delete-orphan")


class ExamQuestion(Base):
    __tablename__ = "exam_questions"
    __table_args__ = (UniqueConstraint("exam_id", "question_id", name="uq_exam_question"),)

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("question_bank_items.id"), nullable=False)
    order_no = Column(Integer, default=1, nullable=False)

    exam = relationship("Exam", back_populates="questions")
    question = relationship("QuestionBankItem")


class ExamAttemptStatus:
    STARTED = "started"
    SUBMITTED = "submitted"
    GRADED = "graded"


class ExamAttempt(Base):
    __tablename__ = "exam_attempts"
    __table_args__ = (UniqueConstraint("exam_id", "user_id", name="uq_exam_user_attempt"),)

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(String(30), default=ExamAttemptStatus.STARTED, nullable=False)
    started_at = Column(DateTime, default=utc_now)
    submitted_at = Column(DateTime, nullable=True)
    graded_at = Column(DateTime, nullable=True)

    total_score = Column(Float, nullable=True)

    exam = relationship("Exam")
    user = relationship("User", foreign_keys=[user_id])
    answers = relationship("ExamAnswer", back_populates="attempt", cascade="all, delete-orphan")


class ExamAnswer(Base):
    __tablename__ = "exam_answers"
    __table_args__ = (UniqueConstraint("attempt_id", "question_id", name="uq_attempt_question"),)

    id = Column(Integer, primary_key=True, index=True)
    attempt_id = Column(Integer, ForeignKey("exam_attempts.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("question_bank_items.id"), nullable=False)

    # 用户答案：single/true_false -> "A"/"true"，multiple -> ["A","C"]
    answer = Column(JSON, nullable=False)

    is_correct = Column(Boolean, nullable=True)
    earned_score = Column(Float, nullable=True)

    attempt = relationship("ExamAttempt", back_populates="answers")
    question = relationship("QuestionBankItem")

