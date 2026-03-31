from enum import Enum
from typing import List
from fastapi import HTTPException, status


class Permission(Enum):
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_USERS = "manage_users"
    MANAGE_TEACHERS = "manage_teachers"
    MANAGE_STUDENTS = "manage_students"
    ALLOCATE_RESOURCES = "allocate_resources"
    VIEW_STUDENTS = "view_students"
    MANAGE_STUDENTS_TEACHER = "manage_students_teacher"
    SEND_FILES = "send_files"
    RECEIVE_FILES = "receive_files"
    GRADE_REPORTS = "grade_reports"
    SUBMIT_REPORTS = "submit_reports"
    MANAGE_KNOWLEDGE_BASE = "manage_knowledge_base"
    VIEW_KNOWLEDGE_BASE = "view_knowledge_base"

    # ---------- 竞赛报名系统 ----------
    VIEW_COMPETITIONS = "view_competitions"
    ENROLL_COMPETITIONS = "enroll_competitions"
    MANAGE_TEAMS = "manage_teams"
    SUBMIT_SUBMISSIONS = "submit_submissions"
    REVIEW_SUBMISSIONS = "review_submissions"
    MANAGE_COMPETITIONS = "manage_competitions"
    PUBLISH_WINNERS = "publish_winners"
    INVIGILATE_EXAMS = "invigilate_exams"

    # ---------- 考试模块 ----------
    MANAGE_QUESTION_BANK = "manage_question_bank"
    MANAGE_EXAMS = "manage_exams"
    TAKE_EXAMS = "take_exams"
    VIEW_EXAM_RESULTS = "view_exam_results"


ROLE_PERMISSIONS = {
    "super_admin": [
        Permission.VIEW_SYSTEM_METRICS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_TEACHERS,
        Permission.MANAGE_STUDENTS,
        Permission.ALLOCATE_RESOURCES,
        Permission.VIEW_STUDENTS,
        Permission.MANAGE_STUDENTS_TEACHER,
        Permission.SEND_FILES,
        Permission.RECEIVE_FILES,
        Permission.GRADE_REPORTS,
        Permission.SUBMIT_REPORTS,
        Permission.MANAGE_KNOWLEDGE_BASE,
        Permission.VIEW_KNOWLEDGE_BASE,

        Permission.VIEW_COMPETITIONS,
        Permission.ENROLL_COMPETITIONS,
        Permission.MANAGE_TEAMS,
        Permission.SUBMIT_SUBMISSIONS,
        Permission.REVIEW_SUBMISSIONS,
        Permission.MANAGE_COMPETITIONS,
        Permission.PUBLISH_WINNERS,
        Permission.INVIGILATE_EXAMS,
        Permission.MANAGE_QUESTION_BANK,
        Permission.MANAGE_EXAMS,
        Permission.TAKE_EXAMS,
        Permission.VIEW_EXAM_RESULTS,
    ],
    "teacher": [
        Permission.VIEW_SYSTEM_METRICS,
        Permission.MANAGE_STUDENTS_TEACHER,
        Permission.VIEW_STUDENTS,
        Permission.SEND_FILES,
        Permission.RECEIVE_FILES,
        Permission.GRADE_REPORTS,
        Permission.SUBMIT_REPORTS,
        Permission.MANAGE_KNOWLEDGE_BASE,
        Permission.VIEW_KNOWLEDGE_BASE,

        Permission.VIEW_COMPETITIONS,
        Permission.REVIEW_SUBMISSIONS,
        Permission.INVIGILATE_EXAMS,
        Permission.MANAGE_QUESTION_BANK,
        Permission.MANAGE_EXAMS,
        Permission.VIEW_EXAM_RESULTS,
        # teacher 角色是否允许队长操作/报名操作，按你的权限矩阵：
        # 教师/评委：负责评分、审核、查看参赛信息；不负责报名/组队。
    ],
    "student": [
        Permission.RECEIVE_FILES,
        Permission.SUBMIT_REPORTS,
        Permission.VIEW_KNOWLEDGE_BASE,

        Permission.VIEW_COMPETITIONS,
        Permission.ENROLL_COMPETITIONS,
        Permission.MANAGE_TEAMS,
        Permission.SUBMIT_SUBMISSIONS,
        Permission.TAKE_EXAMS,
        Permission.VIEW_EXAM_RESULTS,
    ],
}


def check_permission(user_role: str, required_permission: Permission) -> bool:
    if user_role not in ROLE_PERMISSIONS:
        return False
    
    user_permissions = ROLE_PERMISSIONS[user_role]
    return required_permission in user_permissions


def require_permission(user_role: str, required_permission: Permission):
    if not check_permission(user_role, required_permission):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: {required_permission.value}"
        )
    return True


class RoleChecker:
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user_role: str):
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {user_role} not allowed to access this resource"
            )
        return True
