from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional, Tuple, Union
from app.datetime_utils import utc_now, ensure_utc
import os
import uuid

from app.database import get_db
from app.security import get_current_user
from app.permissions import Permission, require_permission
from app.models.user import User, File as FileModel
from app.models.competition import (
    Competition,
    CompetitionEnrollment,
    CompetitionEnrollmentStatus,
    Team,
    TeamMember,
    TeamStatus,
    Submission,
    SubmissionStatus,
    Review,
)
from app.schemas import (
    CompetitionCreate,
    CompetitionUpdate,
    CompetitionResponse,
    CompetitionEnrollmentCreate,
    CompetitionEnrollmentResponse,
    MyEnrollmentResponse,
    TeamCreate,
    TeamResponse,
    TeamDetailResponse,
    TeamMemberResponse,
    IndividualParticipantItem,
    TeamParticipantDetailResponse,
    TeamMemberWithUserResponse,
    TeamTransferCaptain,
    SubmissionCreate,
    SubmissionCreateWrapped,
    SubmissionResponse,
    ReviewGrade,
    ReviewResponse,
    CompetitionScoreSummaryResponse,
    CompetitionScoreRankingItem,
    CompetitionScoreRankingResponse,
    MyCompetitionScoresResponse,
)

router = APIRouter(prefix="/competitions", tags=["Competition Management"])

SUBMISSION_UPLOAD_DIR = "competition_submissions"
os.makedirs(SUBMISSION_UPLOAD_DIR, exist_ok=True)


def _get_competition(db: Session, competition_id: int) -> Competition:
    competition = db.query(Competition).filter(Competition.id == competition_id).first()
    if not competition:
        raise HTTPException(status_code=404, detail="Competition not found")
    return competition


def _is_enrollment_closed(competition: Competition) -> bool:
    """
    「停止报名」条件（与业务上“锁定竞赛”含义一致）：
    1) 状态为 closed（管理员手动关闭报名）
    2) 到达/超过 end_at（自动停止报名）
    """
    if competition.status == "closed":
        return True
    if competition.end_at is None:
        return False
    return utc_now() >= ensure_utc(competition.end_at)


def _ensure_enrollment_open(competition: Competition) -> None:
    """禁止在已停止报名后：新报名 / 新建队伍 / 加入队伍。"""
    if _is_enrollment_closed(competition):
        raise HTTPException(
            status_code=400,
            detail="Competition enrollment is closed (status closed or past end date).",
        )


def _ensure_competition_allows_submissions(competition: Competition) -> None:
    """
    作品提交：已发布（published）或已锁定报名（closed）均允许已参赛用户继续提交；
    草稿（draft）不允许。
    """
    if competition.status not in ("published", "closed"):
        raise HTTPException(
            status_code=400,
            detail="Competition is not accepting submissions (must be published or closed)",
        )


def _ensure_active_individual_enrollment(db: Session, competition_id: int, user_id: int) -> None:
    """个人赛道提交：必须存在「有效个人报名」（enrolled 且 team_id 为空）。"""
    enr = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.competition_id == competition_id,
            CompetitionEnrollment.student_id == user_id,
            CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
            CompetitionEnrollment.team_id.is_(None),
        )
        .first()
    )
    if not enr:
        raise HTTPException(
            status_code=403,
            detail="Not actively enrolled as individual participant in this competition",
        )


def _individual_sequence_no(db: Session, competition_id: int, enrollment: CompetitionEnrollment) -> int:
    """本竞赛个人赛道内序号（从 1 起，按报名时间、id 稳定排序）。"""
    n = (
        db.query(func.count(CompetitionEnrollment.id))
        .filter(
            CompetitionEnrollment.competition_id == competition_id,
            CompetitionEnrollment.team_id.is_(None),
            CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
            or_(
                CompetitionEnrollment.created_at < enrollment.created_at,
                and_(
                    CompetitionEnrollment.created_at == enrollment.created_at,
                    CompetitionEnrollment.id <= enrollment.id,
                ),
            ),
        )
        .scalar()
    )
    return int(n or 0)


def _team_sequence_no(db: Session, competition_id: int, team: Team) -> int:
    """本竞赛组队赛道内队伍序号（从 1 起，按队伍创建时间、队伍 id 排序）。"""
    n = (
        db.query(func.count(Team.id))
        .filter(
            Team.competition_id == competition_id,
            Team.status == TeamStatus.ACTIVE,
            or_(
                Team.created_at < team.created_at,
                and_(Team.created_at == team.created_at, Team.id <= team.id),
            ),
        )
        .scalar()
    )
    return int(n or 0)


def _ensure_submission_access(db: Session, submission: Submission, current_user: User) -> None:
    """
    作品访问边界：
    - super_admin：可访问全部
    - teacher：需具备 REVIEW_SUBMISSIONS（由 require_permission 保证）后可访问全部
    - student：仅可访问自己提交的作品，或其所在队伍提交的作品
    """
    if current_user.role == "super_admin":
        return

    if current_user.role == "teacher":
        require_permission(current_user.role, Permission.REVIEW_SUBMISSIONS)
        return

    # student
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Access denied")

    if submission.team_id is None:
        if submission.student_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        return

    member = db.query(TeamMember).filter(
        TeamMember.team_id == submission.team_id,
        TeamMember.user_id == current_user.id,
    ).first()
    if not member:
        raise HTTPException(status_code=403, detail="Access denied")


@router.post("/", response_model=CompetitionResponse, status_code=status.HTTP_201_CREATED)
async def create_competition(
    competition: CompetitionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_COMPETITIONS)

    comp = Competition(
        name=competition.name,
        description=competition.description,
        rules_text=competition.rules_text,
        status="draft",
        start_at=competition.start_at,
        end_at=competition.end_at,
        allow_individual=competition.allow_individual,
        allow_team=competition.allow_team,
    )
    db.add(comp)
    db.commit()
    db.refresh(comp)
    return comp


@router.get("/", response_model=List[CompetitionResponse])
async def list_competitions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    return db.query(Competition).order_by(Competition.created_at.desc()).all()


@router.get("/enrollments/me", response_model=List[MyEnrollmentResponse])
async def my_enrollments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查看当前用户报名的所有竞赛（含竞赛详情）"""
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    enrollments = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.student_id == current_user.id,
            CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
        )
        .order_by(CompetitionEnrollment.created_at.desc())
        .all()
    )
    results = []
    for e in enrollments:
        comp = db.query(Competition).filter(Competition.id == e.competition_id).first()
        data = MyEnrollmentResponse.model_validate(e)
        if comp:
            data.competition = CompetitionResponse.model_validate(comp)
        results.append(data)
    return results


@router.put("/{competition_id}/publish", response_model=CompetitionResponse)
async def publish_competition(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_COMPETITIONS)
    competition = _get_competition(db, competition_id)
    competition.status = "published"
    db.commit()
    db.refresh(competition)
    return competition


@router.put("/{competition_id}/lock", response_model=CompetitionResponse)
async def lock_competition(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """停止报名：将竞赛标记为 closed（禁止新报名/新建队伍/加入队伍，不禁止提交与评分等）。"""
    require_permission(current_user.role, Permission.MANAGE_COMPETITIONS)
    competition = _get_competition(db, competition_id)
    competition.status = "closed"
    db.commit()
    db.refresh(competition)
    return competition


@router.put("/{competition_id}", response_model=CompetitionResponse)
async def update_competition(
    competition_id: int,
    payload: CompetitionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_COMPETITIONS)
    competition = _get_competition(db, competition_id)

    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(competition, field, value)

    db.commit()
    db.refresh(competition)
    return competition


@router.delete("/{competition_id}", status_code=status.HTTP_200_OK)
async def delete_competition(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_COMPETITIONS)
    competition = _get_competition(db, competition_id)

    db.query(Review).filter(
        Review.submission_id.in_(
            db.query(Submission.id).filter(Submission.competition_id == competition_id)
        )
    ).delete(synchronize_session=False)
    db.query(Submission).filter(Submission.competition_id == competition_id).delete(synchronize_session=False)
    db.query(CompetitionEnrollment).filter(CompetitionEnrollment.competition_id == competition_id).delete(synchronize_session=False)
    db.query(TeamMember).filter(
        TeamMember.team_id.in_(
            db.query(Team.id).filter(Team.competition_id == competition_id)
        )
    ).delete(synchronize_session=False)
    db.query(Team).filter(Team.competition_id == competition_id).delete(synchronize_session=False)
    db.delete(competition)

    db.commit()
    return {"ok": True, "detail": f"Competition {competition_id} and all related data deleted"}


@router.post("/enroll", response_model=CompetitionEnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_competition(
    enroll: CompetitionEnrollmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.ENROLL_COMPETITIONS)

    competition = _get_competition(db, enroll.competition_id)
    _ensure_enrollment_open(competition)
    if competition.status != "published":
        raise HTTPException(status_code=400, detail="Competition not published")

    if current_user.role != "student":
        # 当前版本仅允许 student 报名
        raise HTTPException(status_code=403, detail="Only students can enroll")

    # 允许个人/队伍模式校验
    is_team = enroll.team_id is not None
    if is_team and not competition.allow_team:
        raise HTTPException(status_code=400, detail="Team enrollment not allowed")
    if (not is_team) and not competition.allow_individual:
        raise HTTPException(status_code=400, detail="Individual enrollment not allowed")

    existing_row = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.competition_id == competition.id,
            CompetitionEnrollment.student_id == current_user.id,
        )
        .first()
    )
    if existing_row and existing_row.status == CompetitionEnrollmentStatus.ENROLLED:
        raise HTTPException(status_code=400, detail="Already enrolled in this competition")

    team: Optional[Team] = None
    if is_team:
        team = db.query(Team).filter(Team.id == enroll.team_id, Team.competition_id == competition.id).first()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found in this competition")

        # 同一队伍：成员表必须存在
        member = db.query(TeamMember).filter(
            TeamMember.team_id == team.id, TeamMember.user_id == current_user.id
        ).first()
        if not member:
            raise HTTPException(status_code=403, detail="User is not a team member")

        is_captain = bool(member.is_captain)
        if existing_row and existing_row.status == CompetitionEnrollmentStatus.WITHDRAWN:
            existing_row.team_id = team.id
            existing_row.is_captain = is_captain
            existing_row.status = CompetitionEnrollmentStatus.ENROLLED
            existing_row.student_no = enroll.student_no
            existing_row.real_name = enroll.real_name
            existing_row.college = enroll.college
            existing_row.grade = enroll.grade
            existing_row.contact = enroll.contact
            enrollment = existing_row
        else:
            enrollment = CompetitionEnrollment(
                competition_id=competition.id,
                student_id=current_user.id,
                team_id=team.id,
                is_captain=is_captain,
                status=CompetitionEnrollmentStatus.ENROLLED,
                student_no=enroll.student_no,
                real_name=enroll.real_name,
                college=enroll.college,
                grade=enroll.grade,
                contact=enroll.contact,
            )
            db.add(enrollment)
    else:
        if existing_row and existing_row.status == CompetitionEnrollmentStatus.WITHDRAWN:
            existing_row.team_id = None
            existing_row.is_captain = False
            existing_row.status = CompetitionEnrollmentStatus.ENROLLED
            existing_row.student_no = enroll.student_no
            existing_row.real_name = enroll.real_name
            existing_row.college = enroll.college
            existing_row.grade = enroll.grade
            existing_row.contact = enroll.contact
            enrollment = existing_row
        else:
            enrollment = CompetitionEnrollment(
                competition_id=competition.id,
                student_id=current_user.id,
                team_id=None,
                is_captain=False,
                status=CompetitionEnrollmentStatus.ENROLLED,
                student_no=enroll.student_no,
                real_name=enroll.real_name,
                college=enroll.college,
                grade=enroll.grade,
                contact=enroll.contact,
            )
            db.add(enrollment)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Enrollment failed: {str(e)}")

    db.refresh(enrollment)
    base = CompetitionEnrollmentResponse.model_validate(enrollment)
    if not is_team:
        seq = _individual_sequence_no(db, competition.id, enrollment)
        return base.model_copy(update={"sequence_no": seq})
    assert team is not None
    seq = _team_sequence_no(db, competition.id, team)
    return base.model_copy(update={"sequence_no": seq})


@router.post("/{competition_id}/withdraw", response_model=CompetitionEnrollmentResponse)
async def withdraw_from_competition(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    参赛学生退赛（取消当前竞赛中的有效报名）。
    - 个人参赛：将报名状态置为 withdrawn
    - 组队参赛：从 team_members 移除；队长若队伍内仍有其他成员，须先转让队长，否则无法退赛；
      若队长为队内唯一成员，则退赛同时解散队伍（team 标记为 disbanded）
    """
    require_permission(current_user.role, Permission.ENROLL_COMPETITIONS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can withdraw")

    competition = _get_competition(db, competition_id)

    enrollment = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.competition_id == competition.id,
            CompetitionEnrollment.student_id == current_user.id,
            CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
        )
        .first()
    )
    if not enrollment:
        raise HTTPException(status_code=404, detail="No active enrollment in this competition")

    if enrollment.team_id is None:
        enrollment.status = CompetitionEnrollmentStatus.WITHDRAWN
        db.commit()
        db.refresh(enrollment)
        return enrollment

    team = (
        db.query(Team)
        .filter(Team.id == enrollment.team_id, Team.competition_id == competition.id)
        .first()
    )
    member = db.query(TeamMember).filter(
        TeamMember.team_id == enrollment.team_id,
        TeamMember.user_id == current_user.id,
    ).first()

    if team is None:
        enrollment.status = CompetitionEnrollmentStatus.WITHDRAWN
        enrollment.is_captain = False
        if member:
            db.delete(member)
        db.commit()
        db.refresh(enrollment)
        return enrollment

    is_captain = team.captain_id == current_user.id or (member is not None and member.is_captain)

    if is_captain and team is not None:
        other_members = (
            db.query(TeamMember)
            .filter(TeamMember.team_id == team.id, TeamMember.user_id != current_user.id)
            .count()
        )
        if other_members > 0:
            raise HTTPException(
                status_code=400,
                detail="Captain must transfer captaincy before withdrawing from the competition",
            )
        if member:
            db.delete(member)
        team.status = TeamStatus.DISBANDED
        enrollment.status = CompetitionEnrollmentStatus.WITHDRAWN
        enrollment.is_captain = False
        db.commit()
        db.refresh(enrollment)
        return enrollment

    if member:
        db.delete(member)
    enrollment.status = CompetitionEnrollmentStatus.WITHDRAWN
    enrollment.is_captain = False
    db.commit()
    db.refresh(enrollment)
    return enrollment


@router.get("/{competition_id}/teams", response_model=List[TeamDetailResponse])
async def list_teams(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """查看某竞赛下所有队伍（含成员列表）"""
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    _get_competition(db, competition_id)

    teams = (
        db.query(Team)
        .filter(Team.competition_id == competition_id, Team.status == TeamStatus.ACTIVE)
        .order_by(Team.created_at.desc())
        .all()
    )
    return teams


@router.get("/{competition_id}/participants/individual", response_model=List[IndividualParticipantItem])
async def list_individual_participants(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查看某竞赛「个人赛道」全部有效报名者（`team_id` 为空、`enrolled`）。
    `sequence_no` 为本竞赛个人赛道内序号（从 1 起）；`enrollment_id` 为数据库主键。
    """
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    _get_competition(db, competition_id)

    rows = (
        db.query(CompetitionEnrollment, User)
        .join(User, User.id == CompetitionEnrollment.student_id)
        .filter(
            CompetitionEnrollment.competition_id == competition_id,
            CompetitionEnrollment.team_id.is_(None),
            CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
        )
        .order_by(CompetitionEnrollment.created_at.asc(), CompetitionEnrollment.id.asc())
        .all()
    )
    return [
        IndividualParticipantItem(
            sequence_no=seq,
            enrollment_id=enr.id,
            student_id=enr.student_id,
            username=user.username,
            full_name=user.full_name,
            student_no=enr.student_no,
            real_name=enr.real_name,
            college=enr.college,
            grade=enr.grade,
            contact=enr.contact,
            status=enr.status,
            created_at=enr.created_at,
        )
        for seq, (enr, user) in enumerate(rows, start=1)
    ]


@router.get("/{competition_id}/participants/teams", response_model=List[TeamParticipantDetailResponse])
async def list_team_participants(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    查看某竞赛「组队赛道」全部活跃队伍及成员（含账号名）。
    `sequence_no` 为本竞赛内队伍序号（从 1 起）；队伍的 `id` 仍为全局队伍主键。
    """
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    _get_competition(db, competition_id)

    teams = (
        db.query(Team)
        .options(joinedload(Team.members))
        .filter(Team.competition_id == competition_id, Team.status == TeamStatus.ACTIVE)
        .order_by(Team.created_at.asc(), Team.id.asc())
        .all()
    )
    all_uids = {m.user_id for t in teams for m in t.members}
    users_by_id = {u.id: u for u in db.query(User).filter(User.id.in_(all_uids)).all()} if all_uids else {}

    out: List[TeamParticipantDetailResponse] = []
    for seq, team in enumerate(teams, start=1):
        members_out: List[TeamMemberWithUserResponse] = []
        for m in sorted(team.members, key=lambda x: (x.joined_at or utc_now(), x.id)):
            u = users_by_id.get(m.user_id)
            members_out.append(
                TeamMemberWithUserResponse(
                    id=m.id,
                    team_id=m.team_id,
                    user_id=m.user_id,
                    username=u.username if u else "",
                    full_name=u.full_name if u else None,
                    is_captain=m.is_captain,
                    joined_at=m.joined_at,
                )
            )
        out.append(
            TeamParticipantDetailResponse(
                sequence_no=seq,
                id=team.id,
                competition_id=team.competition_id,
                captain_id=team.captain_id,
                status=team.status,
                created_at=team.created_at,
                members=members_out,
            )
        )
    return out


@router.post("/teams", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    team_create: TeamCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_TEAMS)

    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can create teams")

    competition = _get_competition(db, team_create.competition_id)
    _ensure_enrollment_open(competition)
    if competition.status != "published":
        raise HTTPException(status_code=400, detail="Competition not published")
    if not competition.allow_team:
        raise HTTPException(status_code=400, detail="Team enrollment not allowed")

    existing = db.query(CompetitionEnrollment).filter(
        CompetitionEnrollment.competition_id == competition.id,
        CompetitionEnrollment.student_id == current_user.id,
        CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
    ).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="You have already enrolled in this competition. Each student can only participate once per competition.",
        )

    # 创建队伍并把创建者设为队长
    team = Team(competition_id=competition.id, captain_id=current_user.id, status=TeamStatus.ACTIVE)
    db.add(team)
    db.flush()  # 获取 team.id

    captain_member = TeamMember(team_id=team.id, user_id=current_user.id, is_captain=True)
    db.add(captain_member)

    row_any = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.competition_id == competition.id,
            CompetitionEnrollment.student_id == current_user.id,
        )
        .first()
    )
    if row_any and row_any.status == CompetitionEnrollmentStatus.WITHDRAWN:
        row_any.team_id = team.id
        row_any.is_captain = True
        row_any.status = CompetitionEnrollmentStatus.ENROLLED
    else:
        enrollment = CompetitionEnrollment(
            competition_id=competition.id,
            student_id=current_user.id,
            team_id=team.id,
            is_captain=True,
            status=CompetitionEnrollmentStatus.ENROLLED,
        )
        db.add(enrollment)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Create team failed: {str(e)}")

    db.refresh(team)
    return team


@router.post("/teams/{team_id}/members", response_model=TeamMemberResponse, status_code=status.HTTP_201_CREATED)
async def join_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_TEAMS)

    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can join teams")

    team = db.query(Team).filter(Team.id == team_id).first()
    if not team or team.status != TeamStatus.ACTIVE:
        raise HTTPException(status_code=404, detail="Team not found")

    competition = team.competition
    if not competition or competition.status != "published":
        raise HTTPException(status_code=400, detail="Competition not published")
    _ensure_enrollment_open(competition)

    member = db.query(TeamMember).filter(TeamMember.team_id == team.id, TeamMember.user_id == current_user.id).first()
    if member:
        raise HTTPException(status_code=400, detail="Already a team member")

    existing_enroll = db.query(CompetitionEnrollment).filter(
        CompetitionEnrollment.competition_id == competition.id,
        CompetitionEnrollment.student_id == current_user.id,
        CompetitionEnrollment.status == CompetitionEnrollmentStatus.ENROLLED,
    ).first()
    if existing_enroll:
        raise HTTPException(
            status_code=400,
            detail="You have already enrolled in this competition. Each student can only participate once per competition.",
        )

    member = TeamMember(team_id=team.id, user_id=current_user.id, is_captain=False)
    db.add(member)

    row_any = (
        db.query(CompetitionEnrollment)
        .filter(
            CompetitionEnrollment.competition_id == competition.id,
            CompetitionEnrollment.student_id == current_user.id,
        )
        .first()
    )
    if row_any and row_any.status == CompetitionEnrollmentStatus.WITHDRAWN:
        row_any.team_id = team.id
        row_any.is_captain = False
        row_any.status = CompetitionEnrollmentStatus.ENROLLED
    else:
        enrollment = CompetitionEnrollment(
            competition_id=competition.id,
            student_id=current_user.id,
            team_id=team.id,
            is_captain=False,
            status=CompetitionEnrollmentStatus.ENROLLED,
        )
        db.add(enrollment)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Join team failed: {str(e)}")

    db.refresh(member)
    return member


@router.post("/teams/{team_id}/transfer-captain", response_model=TeamResponse)
async def transfer_captain(
    team_id: int,
    payload: TeamTransferCaptain,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.MANAGE_TEAMS)

    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can transfer captain")

    team = db.query(Team).filter(Team.id == team_id, Team.status == TeamStatus.ACTIVE).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    if team.captain_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only current captain can transfer")

    new_captain_member = db.query(TeamMember).filter(
        TeamMember.team_id == team.id,
        TeamMember.user_id == payload.new_captain_id,
    ).first()
    if not new_captain_member:
        raise HTTPException(status_code=404, detail="New captain must be a team member")

    # 一致性：captain_id + team_members.is_captain
    old_captain_id = team.captain_id
    team.captain_id = payload.new_captain_id

    db.query(TeamMember).filter(TeamMember.team_id == team.id, TeamMember.is_captain == True).update(  # noqa: E712
        {"is_captain": False}
    )
    new_captain_member.is_captain = True

    # 同步 enrollment.is_captain（这也是“队长身份”在报名维度上的一致性来源）
    db.query(CompetitionEnrollment).filter(
        CompetitionEnrollment.competition_id == team.competition_id,
        CompetitionEnrollment.team_id == team.id,
        CompetitionEnrollment.student_id == old_captain_id,
    ).update({"is_captain": False})
    db.query(CompetitionEnrollment).filter(
        CompetitionEnrollment.competition_id == team.competition_id,
        CompetitionEnrollment.team_id == team.id,
        CompetitionEnrollment.student_id == payload.new_captain_id,
    ).update({"is_captain": True})

    db.commit()
    db.refresh(team)
    return team


@router.post("/teams/{team_id}/leave", status_code=status.HTTP_200_OK)
async def leave_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    队长退队：强制先转让。
    - 如果 current_user 是队长，则要求其先通过 transfer 接受“等价的队长身份转移”
    - 允许直接退队后成员仍可继续提交（本接口只处理“退队/移除成员”）
    """
    require_permission(current_user.role, Permission.MANAGE_TEAMS)

    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can leave team")

    team = db.query(Team).filter(Team.id == team_id, Team.status == TeamStatus.ACTIVE).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")

    member = db.query(TeamMember).filter(TeamMember.team_id == team.id, TeamMember.user_id == current_user.id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    if member.is_captain or team.captain_id == current_user.id:
        raise HTTPException(status_code=400, detail="Captain must transfer before leaving")

    db.delete(member)

    # 更新报名状态（同一竞赛同一学生/同一队伍一条报名）
    enrollment = db.query(CompetitionEnrollment).filter(
        CompetitionEnrollment.competition_id == team.competition_id,
        CompetitionEnrollment.team_id == team.id,
        CompetitionEnrollment.student_id == current_user.id,
    ).first()
    if enrollment:
        enrollment.status = CompetitionEnrollmentStatus.WITHDRAWN

    db.commit()
    return {"ok": True}


@router.post("/submissions", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
async def create_submission(
    body: Union[SubmissionCreate, SubmissionCreateWrapped],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    仅接受 `application/json`。需要上传文件请使用 `POST /competitions/submissions/upload`。
    请求体可为 **扁平** `SubmissionCreate`，或 `{"payload": { ...同上字段... }}`（兼容易与 multipart 字段名混淆的前端）。
    若与本接口混用 `UploadFile`，FastAPI 会强制 multipart 并要求名为 `payload` 的字段，导致纯 JSON 返回 422。
    """
    payload = body.payload if isinstance(body, SubmissionCreateWrapped) else body

    require_permission(current_user.role, Permission.SUBMIT_SUBMISSIONS)

    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can submit")

    competition = _get_competition(db, payload.competition_id)
    _ensure_competition_allows_submissions(competition)

    # 校验提交目标：个人 or 队伍
    team_id = payload.team_id
    if team_id is None:
        # 个人提交：student_id 必须是当前用户，且个人报名仍有效
        student_id = current_user.id
        _ensure_active_individual_enrollment(db, competition.id, current_user.id)
    else:
        team = db.query(Team).filter(Team.id == team_id, Team.competition_id == competition.id).first()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        # 队伍提交：提交者必须是队伍成员
        member = db.query(TeamMember).filter(TeamMember.team_id == team.id, TeamMember.user_id == current_user.id).first()
        if not member:
            raise HTTPException(status_code=403, detail="User is not a team member")
        student_id = current_user.id

    if not payload.file_id and not payload.content_text:
        raise HTTPException(status_code=400, detail="Provide file_id or content_text")

    file_id = payload.file_id
    content_text = payload.content_text

    submission = Submission(
        competition_id=competition.id,
        team_id=payload.team_id,
        student_id=student_id,
        submitter_id=current_user.id,
        title=payload.title,
        description=payload.description,
        file_id=file_id,
        content_text=content_text,
        status=SubmissionStatus.SUBMITTED,
    )
    db.add(submission)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Create submission failed: {str(e)}")

    db.refresh(submission)
    return submission


@router.post("/submissions/upload", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
async def create_submission_upload(
    competition_id: int = Form(...),
    team_id: Optional[int] = Form(None),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    content_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    multipart/form-data 方式提交作品（支持文件上传）。
    """
    require_permission(current_user.role, Permission.SUBMIT_SUBMISSIONS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can submit")

    competition = _get_competition(db, competition_id)
    _ensure_competition_allows_submissions(competition)

    # 校验提交目标：个人 or 队伍
    if team_id is None:
        student_id = current_user.id
        _ensure_active_individual_enrollment(db, competition.id, current_user.id)
    else:
        team = db.query(Team).filter(Team.id == team_id, Team.competition_id == competition.id).first()
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        member = db.query(TeamMember).filter(TeamMember.team_id == team.id, TeamMember.user_id == current_user.id).first()
        if not member:
            raise HTTPException(status_code=403, detail="User is not a team member")
        student_id = current_user.id

    if not content_text and not (file and file.filename):
        raise HTTPException(status_code=400, detail="Provide content_text or upload file")

    file_id = None
    if file is not None and file.filename:
        file_uuid = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        filename = f"{file_uuid}{ext}"
        file_path = os.path.join(SUBMISSION_UPLOAD_DIR, filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        file_record = FileModel(
            filename=file.filename,
            file_type="submission",
            file_path=file_path,
            file_size=len(content),
            mime_type=file.content_type,
            sender_id=current_user.id,
        )
        db.add(file_record)
        db.flush()
        file_id = file_record.id

    submission = Submission(
        competition_id=competition.id,
        team_id=team_id,
        student_id=student_id,
        submitter_id=current_user.id,
        title=title,
        description=description,
        file_id=file_id,
        content_text=content_text,
        status=SubmissionStatus.SUBMITTED,
    )
    db.add(submission)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Create submission failed: {str(e)}")

    db.refresh(submission)
    return submission


@router.get("/{competition_id}/submissions", response_model=List[SubmissionResponse])
async def list_submissions(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    _get_competition(db, competition_id)

    if current_user.role in ["teacher", "super_admin"]:
        # teacher/super_admin 可查看竞赛内全部提交（teacher 需要 REVIEW_SUBMISSIONS）
        if current_user.role == "teacher":
            require_permission(current_user.role, Permission.REVIEW_SUBMISSIONS)
        return db.query(Submission).filter(Submission.competition_id == competition_id).order_by(Submission.submitted_at.desc()).all()

    # student：仅看自己的个人提交 + 自己所在队伍的提交
    team_ids = [tm.team_id for tm in db.query(TeamMember).join(Team).filter(
        TeamMember.user_id == current_user.id,
        Team.competition_id == competition_id,
    ).all()]

    q = db.query(Submission).filter(Submission.competition_id == competition_id).filter(
        (Submission.student_id == current_user.id) | (Submission.team_id.in_(team_ids) if team_ids else False)  # noqa: E712
    )
    return q.order_by(Submission.submitted_at.desc()).all()


@router.get("/submissions/{submission_id}", response_model=SubmissionResponse)
async def get_submission(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    _ensure_submission_access(db, submission, current_user)
    return submission


@router.get("/submissions/{submission_id}/download")
async def download_submission_file(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    _ensure_submission_access(db, submission, current_user)

    if not submission.file_id:
        raise HTTPException(status_code=404, detail="No file attached")

    file_record = db.query(FileModel).filter(FileModel.id == submission.file_id).first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    if not file_record.file_path or not os.path.exists(file_record.file_path):
        raise HTTPException(status_code=404, detail="File missing on server")

    return FileResponse(
        path=file_record.file_path,
        filename=file_record.filename,
        media_type=file_record.mime_type or "application/octet-stream",
    )


@router.put("/submissions/{submission_id}/review-grade", response_model=ReviewResponse)
async def review_submission(
    submission_id: int,
    grade: ReviewGrade,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    require_permission(current_user.role, Permission.REVIEW_SUBMISSIONS)
    if current_user.role not in ["teacher", "super_admin"]:
        raise HTTPException(status_code=403, detail="Only teacher/super_admin can review")

    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.review is not None:
        raise HTTPException(status_code=400, detail="Submission already reviewed")

    review = Review(
        submission_id=submission.id,
        reviewer_id=current_user.id,
        status=SubmissionStatus.APPROVED,
        score=grade.score,
        feedback=grade.feedback,
        reviewed_at=utc_now(),
    )
    db.add(review)
    submission.status = SubmissionStatus.APPROVED
    db.commit()
    db.refresh(review)
    return review


@router.get("/{competition_id}/scores/summary", response_model=CompetitionScoreSummaryResponse)
async def score_summary(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    竞赛评分汇总（聚合统计）。
    - super_admin / teacher(评委) 可查看全竞赛
    """
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    require_permission(current_user.role, Permission.REVIEW_SUBMISSIONS)
    _get_competition(db, competition_id)

    submissions_total = db.query(func.count(Submission.id)).filter(Submission.competition_id == competition_id).scalar() or 0
    reviewed_total = (
        db.query(func.count(Review.id))
        .join(Submission, Submission.id == Review.submission_id)
        .filter(Submission.competition_id == competition_id)
        .scalar()
        or 0
    )

    agg = (
        db.query(
            func.avg(Review.score),
            func.max(Review.score),
            func.min(Review.score),
        )
        .join(Submission, Submission.id == Review.submission_id)
        .filter(Submission.competition_id == competition_id)
        .first()
    )

    avg_score = float(agg[0]) if agg and agg[0] is not None else None
    max_score = float(agg[1]) if agg and agg[1] is not None else None
    min_score = float(agg[2]) if agg and agg[2] is not None else None

    return CompetitionScoreSummaryResponse(
        competition_id=competition_id,
        submissions_total=int(submissions_total),
        reviewed_total=int(reviewed_total),
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
    )


@router.get("/{competition_id}/scores/rankings", response_model=CompetitionScoreRankingResponse)
async def score_rankings(
    competition_id: int,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    排行榜：个人参赛与组队参赛**同一排名池**，按各参赛者（队伍或个人）的 `best_score` 全局排序。
    - 先分别聚合「有评分作品」的队伍与个人，再合并排序；**不在**各自赛道单独截断 limit（避免名次被截断错误）。
    - `limit` 仅作用于合并排序后的最终结果条数。
    """
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    require_permission(current_user.role, Permission.REVIEW_SUBMISSIONS)
    _get_competition(db, competition_id)

    # 队伍参赛：按 team_id 聚合（本竞赛内有已评分作品的所有队伍）
    team_rows = (
        db.query(
            Submission.team_id.label("team_id"),
            func.max(Review.score).label("best_score"),
            func.count(Review.id).label("reviewed_submissions"),
        )
        .join(Review, Review.submission_id == Submission.id)
        .filter(Submission.competition_id == competition_id, Submission.team_id.isnot(None))
        .group_by(Submission.team_id)
        .all()
    )

    # 个人参赛：按 student_id 聚合（team_id 为空）
    individual_rows = (
        db.query(
            Submission.student_id.label("student_id"),
            func.max(Review.score).label("best_score"),
            func.count(Review.id).label("reviewed_submissions"),
        )
        .join(Review, Review.submission_id == Submission.id)
        .filter(Submission.competition_id == competition_id, Submission.team_id.is_(None))
        .group_by(Submission.student_id)
        .all()
    )

    # 合并为统一列表后再排序（同分按队伍优先、再按 id 稳定次序）
    pool: List[Tuple[Optional[int], Optional[int], float, int]] = []
    for r in team_rows:
        pool.append((int(r.team_id), None, float(r.best_score), int(r.reviewed_submissions)))
    for r in individual_rows:
        pool.append((None, int(r.student_id), float(r.best_score), int(r.reviewed_submissions)))

    pool.sort(key=lambda x: (-x[2], 0 if x[0] is not None else 1, x[0] or 0, x[1] or 0))

    items: List[CompetitionScoreRankingItem] = []
    rank_val = 1
    for i, (tid, sid, best, rcnt) in enumerate(pool):
        if i > 0 and best < pool[i - 1][2]:
            rank_val = i + 1
        items.append(
            CompetitionScoreRankingItem(
                rank=rank_val,
                team_id=tid,
                student_id=sid,
                best_score=best,
                reviewed_submissions=rcnt,
            )
        )

    return CompetitionScoreRankingResponse(competition_id=competition_id, items=items[:limit])


@router.get("/{competition_id}/scores/me", response_model=MyCompetitionScoresResponse)
async def my_scores(
    competition_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    学生查看自己在某竞赛的提交与成绩。
    """
    require_permission(current_user.role, Permission.VIEW_COMPETITIONS)
    if current_user.role != "student":
        raise HTTPException(status_code=403, detail="Only students can view my scores")

    _get_competition(db, competition_id)
    # 自己个人提交 + 自己所在队伍提交
    team_ids = [tm.team_id for tm in db.query(TeamMember).join(Team).filter(
        TeamMember.user_id == current_user.id,
        Team.competition_id == competition_id,
    ).all()]

    q = db.query(Submission).filter(Submission.competition_id == competition_id).filter(
        (Submission.student_id == current_user.id) | (Submission.team_id.in_(team_ids) if team_ids else False)  # noqa: E712
    )
    submissions = q.order_by(Submission.submitted_at.desc()).all()
    return MyCompetitionScoresResponse(competition_id=competition_id, submissions=submissions)

