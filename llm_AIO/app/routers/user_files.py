import os
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models.user import User, File as FileModel
from app.schemas import FileCreate, FileResponse, BatchFileSend, UserResponse
from app.security import get_current_user
from app.permissions import require_permission, Permission

router = APIRouter(prefix="/user-files", tags=["User File Management"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = "material",
    description: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.SEND_FILES)
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    filename = f"{file_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    file_record = FileModel(
        filename=file.filename,
        file_type=file_type,
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type,
        sender_id=current_user.id,
        description=description
    )
    
    db.add(file_record)
    db.commit()
    db.refresh(file_record)
    
    return file_record


def _parse_receiver_id(value: Optional[str]) -> Optional[int]:
    """将 Form 传来的 receiver_id（多为字符串）转为 int，multipart 下更可靠"""
    if value is None:
        return None
    s = (value if isinstance(value, str) else str(value)).strip()
    if not s:
        return None
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


@router.post("/send", response_model=List[FileResponse])
async def send_file(
    file: UploadFile = File(...),
    receiver_id: Optional[str] = Form(None),  # multipart 下多为字符串，在函数内转为 int
    batch_data_json: Optional[str] = Form(None),  # 批量发送时传此 Form 字段，值为 JSON 字符串
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """个人发送：传 receiver_id；批量发送：传 batch_data（Form 字段，值为 JSON 字符串）"""
    require_permission(current_user.role, Permission.SEND_FILES)
    
    batch_obj = None
    if batch_data_json:
        try:
            batch_obj = BatchFileSend(**json.loads(batch_data_json))
        except (json.JSONDecodeError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid batch_data JSON")
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ""
    filename = f"{file_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 个人发送：仅 receiver_id；批量发送：batch_data.receiver_ids
    is_batch_send = batch_obj and batch_obj.receiver_ids
    single_receiver_id = None
    if not is_batch_send:
        single_receiver_id = _parse_receiver_id(receiver_id)

    file_record = FileModel(
        filename=file.filename,
        file_type=batch_obj.file_type if batch_obj else "material",
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type,
        sender_id=current_user.id,
        receiver_id=single_receiver_id,  # 个人发送时指定接收者
        description=batch_obj.description if batch_obj else None,
        is_batch=bool(is_batch_send),
        batch_group_id=str(uuid.uuid4()) if is_batch_send else None
    )
    
    db.add(file_record)
    db.flush()
    
    if is_batch_send:
        for rid in batch_obj.receiver_ids:
            receiver = db.query(User).filter(User.id == rid).first()
            if not receiver:
                continue
            
            file_copy = FileModel(
                filename=file.filename,
                file_type=batch_obj.file_type,
                file_path=file_path,
                file_size=len(content),
                mime_type=file.content_type,
                sender_id=current_user.id,
                receiver_id=rid,
                description=batch_obj.description,
                is_batch=True,
                batch_group_id=file_record.batch_group_id
            )
            db.add(file_copy)
    
    db.commit()
    
    return [file_record]


@router.get("/received", response_model=List[FileResponse])
async def get_received_files(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    files = db.query(FileModel).filter(
        FileModel.receiver_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return files


@router.get("/sent", response_model=List[FileResponse])
async def get_sent_files(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    files = db.query(FileModel).filter(
        FileModel.sender_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return files


@router.get("/student/{student_id}", response_model=List[FileResponse])
async def get_student_files(
    student_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_STUDENTS)
    
    files = db.query(FileModel).filter(
        FileModel.receiver_id == student_id
    ).offset(skip).limit(limit).all()
    
    return files


@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    if (current_user.role == "student" and 
        current_user.id != file_record.receiver_id and
        current_user.id != file_record.sender_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return file_record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    file_record = db.query(FileModel).filter(FileModel.id == file_id).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    if current_user.role != "super_admin" and current_user.id != file_record.sender_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete files sent by others"
        )
    
    if os.path.exists(file_record.file_path):
        os.remove(file_record.file_path)
    
    db.delete(file_record)
    db.commit()
