from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Float, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import UserBase as Base
from app.datetime_utils import utc_now


class CompetitionStatus(str):
    # 使用 str 常量避免额外枚举依赖；可按需替换为 enum.Enum
    DRAFT = "draft"
    PUBLISHED = "published"
    CLOSED = "closed"


class Competition(Base):
    __tablename__ = "competitions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    rules_text = Column(Text)

    status = Column(String(30), default=CompetitionStatus.DRAFT, nullable=False)
    start_at = Column(DateTime, nullable=True)
    end_at = Column(DateTime, nullable=True)

    # 一个竞赛内：可以配置为仅团队/仅个人/都允许
    allow_individual = Column(Boolean, default=True, nullable=False)
    allow_team = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    teams = relationship("Team", back_populates="competition")
    enrollments = relationship("CompetitionEnrollment", back_populates="competition")


class CompetitionEnrollmentStatus(str):
    ENROLLED = "enrolled"
    WITHDRAWN = "withdrawn"


class CompetitionEnrollment(Base):
    """
    表示“学生在某个竞赛中的报名记录”。
    - 支持个人参赛：team_id 为空
    - 支持组队参赛：team_id 不为空
    """

    __tablename__ = "competition_enrollments"
    __table_args__ = (
        UniqueConstraint("competition_id", "student_id", name="uq_competition_student"),
    )

    id = Column(Integer, primary_key=True, index=True)

    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # team_id nullable：允许“个人参赛与队伍参赛共存”
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)

    # 记录学生在竞赛中的身份（仅用于查询/审计）
    is_captain = Column(Boolean, default=False, nullable=False)

    # 报名时填写的参赛学生信息
    student_no = Column(String(50), nullable=True, comment="学号")
    real_name = Column(String(100), nullable=True, comment="姓名")
    college = Column(String(200), nullable=True, comment="学院")
    grade = Column(String(50), nullable=True, comment="年级，如 2023级")
    contact = Column(String(100), nullable=True, comment="联系方式（手机/邮箱）")

    status = Column(String(30), default=CompetitionEnrollmentStatus.ENROLLED, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    competition = relationship("Competition", back_populates="enrollments")
    student = relationship("User", foreign_keys=[student_id])
    # 注意：Enrollment.team 关联到的是“队伍”本身；
    # Team.members 是 TeamMember 的关系，两者不需要 back_populates 互指，否则会造成关系反向映射冲突。
    team = relationship("Team", foreign_keys=[team_id], uselist=False)


class TeamStatus(str):
    ACTIVE = "active"
    DISBANDED = "disbanded"


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)

    # 房间内队长（强一致性由路由逻辑保证）
    captain_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(String(30), default=TeamStatus.ACTIVE, nullable=False)
    created_at = Column(DateTime, default=utc_now)

    competition = relationship("Competition", back_populates="teams")
    members = relationship("TeamMember", back_populates="team", cascade="all, delete-orphan")

    # captain 关系（方便序列化时使用；路由里也可直接用 captain_id）
    captain = relationship("User", foreign_keys=[captain_id])


class TeamMember(Base):
    __tablename__ = "team_members"
    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_member"),
    )

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    is_captain = Column(Boolean, default=False, nullable=False)
    joined_at = Column(DateTime, default=utc_now)

    team = relationship("Team", back_populates="members")
    user = relationship("User", foreign_keys=[user_id])


class SubmissionStatus(str):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class Submission(Base):
    __tablename__ = "submissions"
    __table_args__ = (
        # 可选：同一个队伍/个人在同一竞赛只允许一个“当前提交”，
        # 如需历史版本可移除该约束改成版本号。
        UniqueConstraint("competition_id", "team_id", "student_id", name="uq_submission_competition_target"),
    )

    id = Column(Integer, primary_key=True, index=True)

    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False)

    # 提交对象：
    # - 若 team_id 非空，则代表队伍提交（队伍成员共同拥有作品）
    # - 若 team_id 为空，则代表个人提交（student_id 填写）
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    submitter_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    title = Column(String(200), nullable=False)
    description = Column(Text)

    # 作品内容：优先落文件（复用现有 File 模型更省事）
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    content_text = Column(Text, nullable=True)

    status = Column(String(30), default=SubmissionStatus.SUBMITTED, nullable=False)
    submitted_at = Column(DateTime, default=utc_now)

    competition = relationship("Competition")
    team = relationship("Team")
    student = relationship("User", foreign_keys=[student_id])
    submitter = relationship("User", foreign_keys=[submitter_id])
    review = relationship("Review", back_populates="submission", uselist=False, cascade="all, delete-orphan")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False, unique=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 审核/评分字段
    status = Column(String(30), default=SubmissionStatus.UNDER_REVIEW, nullable=False)
    score = Column(Float, nullable=True)
    feedback = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    submission = relationship("Submission", back_populates="review")
    reviewer = relationship("User", foreign_keys=[reviewer_id])


# 由于本仓库的 User / File 定义在 app.models.user，避免循环导入：
# 使用字符串形式关系已经足够；但在某些数据库/序列化工具下，
# 需要保证这些模型在 init_db 中被 import 才能创建表。

