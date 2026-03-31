from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.datetime_utils import utc_now
import uuid
import os

from app.database import get_db
from app.models.user import User, Report, File as FileModel
from app.schemas import ReportCreate, ReportResponse, ReportGrade, UserResponse
from app.security import get_current_user
from app.permissions import require_permission, Permission

router = APIRouter(prefix="/reports", tags=["Report Management"])

REPORT_UPLOAD_DIR = "reports"
os.makedirs(REPORT_UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def submit_report(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.SUBMIT_REPORTS)
    
    if current_user.role != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can submit reports"
        )
    
    file_id = None
    file_path = None
    file_record = None
    
    try:
        if file and file.filename:
            file_uuid = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            filename = f"{file_uuid}{file_extension}"
            file_path = os.path.join(REPORT_UPLOAD_DIR, filename)
            
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            file_record = FileModel(
                filename=file.filename,
                file_type="submission",
                file_path=file_path,
                file_size=len(content),
                mime_type=file.content_type,
                sender_id=current_user.id
            )
            db.add(file_record)
            db.flush()  # 刷新以获取文件记录的ID
            file_id = file_record.id
        
        report = Report(
            title=title,
            description=description,
            student_id=current_user.id,
            file_id=file_id
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        return report
    except Exception as e:
        # 如果出错，回滚事务
        db.rollback()
        # 如果文件已上传但事务失败，清理文件
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit report: {str(e)}"
        )


@router.get("/my-reports", response_model=List[ReportResponse])
async def get_my_reports(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role == "student":
        reports = db.query(Report).filter(
            Report.student_id == current_user.id
        ).offset(skip).limit(limit).all()
    else:
        reports = db.query(Report).filter(
            Report.teacher_id == current_user.id
        ).offset(skip).limit(limit).all()
    
    return reports


@router.get("/student/{student_id}", response_model=List[ReportResponse])
async def get_student_reports(
    student_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.GRADE_REPORTS)
    
    student = db.query(User).filter(User.id == student_id, User.role == "student").first()
    if not student:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found"
        )
    
    reports = db.query(Report).filter(
        Report.student_id == student_id
    ).offset(skip).limit(limit).all()
    
    return reports


@router.get("/pending", response_model=List[ReportResponse])
async def get_pending_reports(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.GRADE_REPORTS)
    
    reports = db.query(Report).filter(
        Report.status == "submitted"
    ).offset(skip).limit(limit).all()
    
    return reports


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if (current_user.role == "student" and 
        current_user.id != report.student_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other students' reports"
        )
    
    return report


@router.put("/{report_id}/grade", response_model=ReportResponse)
async def grade_report(
    report_id: int,
    grade_data: ReportGrade,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.GRADE_REPORTS)
    
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if report.status == "graded":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Report has already been graded"
        )
    
    report.grade = grade_data.grade
    report.feedback = grade_data.feedback
    report.teacher_id = current_user.id
    report.status = "graded"
    report.graded_at = utc_now()
    
    db.commit()
    db.refresh(report)
    
    return report


@router.delete("/{report_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found"
        )
    
    if current_user.role == "student":
        if current_user.id != report.student_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other students' reports"
            )
        if report.status == "graded":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete graded reports"
            )
    elif current_user.role == "teacher":
        if current_user.id != report.teacher_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot delete other teachers' graded reports"
            )
    
    db.delete(report)
    db.commit()
