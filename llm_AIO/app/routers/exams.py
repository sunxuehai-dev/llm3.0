from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Any
from datetime import timedelta

from app.datetime_utils import utc_now, ensure_utc

from app.database import get_db
from app.security import get_current_user
from app.permissions import Permission, require_permission
from app.models.user import User
from app.models.exam import (
    QuestionBankItem,
    QuestionType,
    Exam,
    ExamStatus,
    ExamQuestion,
    ExamAttempt,
    ExamAttemptStatus,
    ExamAnswer,
)
from app.schemas import (
    QuestionBankItemCreate,
    QuestionBankItemResponse,
    ExamCreate,
    ExamResponse,
    ExamPublishResponse,
    ExamAttemptStartResponse,
    ExamSubmitRequest,
    ExamAttemptResponse,
    ExamAttemptDetailResponse,
)

router = APIRouter(prefix="/exams", tags=["Exam Management"])


def _normalize_answer(question_type: str, answer: Any):
    if question_type == QuestionType.MULTIPLE:
        if not isinstance(answer, list):
            raise HTTPException(status_code=400, detail="Multiple choice answer must be a list")
        return sorted([str(x) for x in answer])
    return str(answer)


def _is_correct(question: QuestionBankItem, user_answer: Any) -> bool:
    qtype = question.question_type
    correct = question.correct_answer
    ua = _normalize_answer(qtype, user_answer)
    ca = _normalize_answer(qtype, correct)
    return ua == ca


def _check_exam_window_for_start(exam: Exam) -> None:
    now = utc_now()
    if exam.start_at and now < ensure_utc(exam.start_at):
        raise HTTPException(status_code=400, detail="Exam not started yet")
    if exam.end_at and now > ensure_utc(exam.end_at):
        raise HTTPException(status_code=400, detail="Exam already ended")


def _check_exam_window_for_submit(exam: Exam, attempt: ExamAttempt) -> None:
    now = utc_now()
    if exam.start_at and now < ensure_utc(exam.start_at):
        raise HTTPException(status_code=400, detail="Exam not started yet")
    if exam.end_at and now > ensure_utc(exam.end_at):
        raise HTTPException(status_code=400, detail="Exam already ended")

    # 时长到期：attempt.started_at + duration_minutes
    if attempt.started_at and exam.duration_minutes:
        started = ensure_utc(attempt.started_at)
        deadline = started + timedelta(minutes=int(exam.duration_minutes)) if started else None
        if deadline is not None and now > deadline:
            raise HTTPException(status_code=400, detail="Exam attempt timed out")


def _grade_attempt(db: Session, attempt: ExamAttempt, answers_payload: Optional[ExamSubmitRequest] = None) -> float:
    """
    自动阅卷并写入 ExamAnswer / attempt.total_score。
    - 若 answers_payload 为 None，则使用 attempt.answers 中已有答案（强制交卷场景）。
    """
    total = 0.0

    if answers_payload is not None:
        for ans in answers_payload.answers:
            q = db.query(QuestionBankItem).filter(QuestionBankItem.id == ans.question_id).first()
            if not q:
                raise HTTPException(status_code=404, detail=f"Question not found: {ans.question_id}")

            correct = _is_correct(q, ans.answer)
            earned = float(q.score) if correct else 0.0
            total += earned

            rec = ExamAnswer(
                attempt_id=attempt.id,
                question_id=q.id,
                answer=ans.answer,
                is_correct=correct,
                earned_score=earned,
            )
            db.add(rec)
    else:
        # 使用已有答案记录重新计算总分（不重复写入答案）
        for rec in attempt.answers:
            q = db.query(QuestionBankItem).filter(QuestionBankItem.id == rec.question_id).first()
            if not q:
                continue
            correct = _is_correct(q, rec.answer)
            earned = float(q.score) if correct else 0.0
            rec.is_correct = correct
            rec.earned_score = earned
            total += earned

    attempt.status = ExamAttemptStatus.GRADED
    attempt.submitted_at = utc_now()
    attempt.graded_at = utc_now()
    attempt.total_score = total
    return total


@router.post("/question-bank", response_model=QuestionBankItemResponse, status_code=status.HTTP_201_CREATED)
async def create_question(
    payload: QuestionBankItemCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_QUESTION_BANK)

    item = QuestionBankItem(
        created_by=current_user.id,
        question_type=payload.question_type,
        stem=payload.stem,
        options=payload.options,
        correct_answer=payload.correct_answer,
        score=payload.score,
        is_active=True,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.get("/question-bank", response_model=List[QuestionBankItemResponse])
async def list_questions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    qtype: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
):
    require_permission(current_user.role, Permission.MANAGE_QUESTION_BANK)
    q = db.query(QuestionBankItem).filter(QuestionBankItem.is_active == True)  # noqa: E712
    if qtype:
        q = q.filter(QuestionBankItem.question_type == qtype)
    return q.order_by(QuestionBankItem.created_at.desc()).offset(skip).limit(limit).all()


@router.post("/", response_model=ExamResponse, status_code=status.HTTP_201_CREATED)
async def create_exam(
    payload: ExamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_EXAMS)

    exam = Exam(
        competition_id=payload.competition_id,
        title=payload.title,
        description=payload.description,
        status=ExamStatus.DRAFT,
        start_at=payload.start_at,
        end_at=payload.end_at,
        duration_minutes=payload.duration_minutes,
        created_by=current_user.id,
    )
    db.add(exam)
    db.flush()

    # 绑定题目
    order_no = 1
    for qid in payload.question_ids:
        qb = db.query(QuestionBankItem).filter(QuestionBankItem.id == qid, QuestionBankItem.is_active == True).first()  # noqa: E712
        if not qb:
            raise HTTPException(status_code=404, detail=f"Question not found: {qid}")
        db.add(ExamQuestion(exam_id=exam.id, question_id=qid, order_no=order_no))
        order_no += 1

    db.commit()
    db.refresh(exam)
    return exam


@router.put("/{exam_id}/publish", response_model=ExamPublishResponse)
async def publish_exam(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_EXAMS)
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    exam.status = ExamStatus.PUBLISHED
    db.commit()
    return {"exam_id": exam.id, "status": exam.status}


@router.get("/", response_model=List[ExamResponse])
async def list_exams(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # 学生也允许浏览已发布考试
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    q = db.query(Exam)
    if current_user.role == "student":
        q = q.filter(Exam.status == ExamStatus.PUBLISHED)
    return q.order_by(Exam.created_at.desc()).all()


@router.post("/{exam_id}/start", response_model=ExamAttemptStartResponse)
async def start_exam(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.TAKE_EXAMS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can take exams")

    exam = db.query(Exam).filter(Exam.id == exam_id, Exam.status == ExamStatus.PUBLISHED).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found or not published")
    _check_exam_window_for_start(exam)

    attempt = db.query(ExamAttempt).filter(ExamAttempt.exam_id == exam_id, ExamAttempt.user_id == current_user.id).first()
    if attempt:
        return {"attempt_id": attempt.id, "status": attempt.status}

    attempt = ExamAttempt(exam_id=exam_id, user_id=current_user.id, status=ExamAttemptStatus.STARTED)
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return {"attempt_id": attempt.id, "status": attempt.status}


@router.post("/{exam_id}/submit", response_model=ExamAttemptResponse)
async def submit_exam(
    exam_id: int,
    payload: ExamSubmitRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.TAKE_EXAMS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can submit exams")

    attempt = db.query(ExamAttempt).filter(
        ExamAttempt.exam_id == exam_id,
        ExamAttempt.user_id == current_user.id,
    ).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found, start exam first")
    if attempt.status in (ExamAttemptStatus.SUBMITTED, ExamAttemptStatus.GRADED):
        raise HTTPException(status_code=400, detail="Attempt already submitted")

    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    _check_exam_window_for_submit(exam, attempt)

    _grade_attempt(db, attempt, answers_payload=payload)

    db.commit()
    db.refresh(attempt)
    return attempt


@router.get("/{exam_id}/attempts/me", response_model=ExamAttemptResponse)
async def get_my_attempt(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_EXAM_RESULTS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can view my attempt")
    attempt = db.query(ExamAttempt).filter(ExamAttempt.exam_id == exam_id, ExamAttempt.user_id == current_user.id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return attempt


@router.get("/{exam_id}/attempts", response_model=List[ExamAttemptResponse])
async def list_attempts(
    exam_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    监考/教师查看某场考试的考生作答情况列表。
    """
    require_permission(current_user.role, Permission.INVIGILATE_EXAMS)
    exam = db.query(Exam).filter(Exam.id == exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")
    return db.query(ExamAttempt).filter(ExamAttempt.exam_id == exam_id).order_by(ExamAttempt.started_at.desc()).all()


@router.get("/attempts/{attempt_id}", response_model=ExamAttemptDetailResponse)
async def get_attempt_detail(
    attempt_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    监考/教师查看某个 attempt 的答案明细与判分。
    """
    require_permission(current_user.role, Permission.INVIGILATE_EXAMS)
    attempt = db.query(ExamAttempt).filter(ExamAttempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    # 触发 answers 关系加载
    _ = attempt.answers
    return attempt


@router.post("/attempts/{attempt_id}/force-submit", response_model=ExamAttemptResponse)
async def force_submit_attempt(
    attempt_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    监考强制交卷：对指定 attempt 强制提交并自动阅卷。
    - 按已存在的答案记录计分（未作答得 0 分）
    """
    require_permission(current_user.role, Permission.INVIGILATE_EXAMS)

    attempt = db.query(ExamAttempt).filter(ExamAttempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt.status == ExamAttemptStatus.GRADED:
        return attempt

    exam = db.query(Exam).filter(Exam.id == attempt.exam_id).first()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    # 强制交卷也遵守考试结束时间（但不强制检查时长；监考可在超时后执行）
    if exam.end_at and utc_now() > ensure_utc(exam.end_at):
        pass

    _grade_attempt(db, attempt, answers_payload=None)
    db.commit()
    db.refresh(attempt)
    return attempt

