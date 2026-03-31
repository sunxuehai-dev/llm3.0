import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.models.user import User, KnowledgeBaseEntry, File as FileModel
from app.schemas import KnowledgeBaseEntryCreate, KnowledgeBaseEntryResponse, UserResponse
from app.security import get_current_user
from app.permissions import require_permission, Permission

router = APIRouter(prefix="/knowledge-base", tags=["Knowledge Base Management"])

KB_UPLOAD_DIR = "knowledge_base"
os.makedirs(KB_UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=KnowledgeBaseEntryResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge_entry(
    title: str = Form(...),
    content: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    
    file_id = None
    if file and file.filename:
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{file_id}{file_extension}"
        file_path = os.path.join(KB_UPLOAD_DIR, filename)
        
        file_content = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        file_record = FileModel(
            filename=file.filename,
            file_type="knowledge_base",
            file_path=file_path,
            file_size=len(file_content),
            mime_type=file.content_type,
            sender_id=current_user.id
        )
        db.add(file_record)
        db.flush()
        file_id = file_record.id
    
    tag_list = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
    
    entry = KnowledgeBaseEntry(
        title=title,
        content=content,
        category=category,
        tags=tag_list,
        file_id=file_id,
        uploader_id=current_user.id
    )
    
    db.add(entry)
    db.commit()
    db.refresh(entry)
    
    return entry


@router.get("/", response_model=List[KnowledgeBaseEntryResponse])
async def get_knowledge_entries(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_KNOWLEDGE_BASE)
    
    query = db.query(KnowledgeBaseEntry)
    
    if category:
        query = query.filter(KnowledgeBaseEntry.category == category)
    
    if search:
        query = query.filter(
            (KnowledgeBaseEntry.title.contains(search)) |
            (KnowledgeBaseEntry.content.contains(search))
        )
    
    entries = query.offset(skip).limit(limit).all()
    
    return entries


@router.get("/categories")
async def get_categories(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_KNOWLEDGE_BASE)
    
    categories = db.query(KnowledgeBaseEntry.category).distinct().all()
    return [c[0] for c in categories if c[0]]


@router.get("/{entry_id}", response_model=KnowledgeBaseEntryResponse)
async def get_knowledge_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    entry = db.query(KnowledgeBaseEntry).filter(KnowledgeBaseEntry.id == entry_id).first()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base entry not found"
        )
    
    return entry


@router.put("/{entry_id}", response_model=KnowledgeBaseEntryResponse)
async def update_knowledge_entry(
    entry_id: int,
    title: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    
    entry = db.query(KnowledgeBaseEntry).filter(KnowledgeBaseEntry.id == entry_id).first()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base entry not found"
        )
    
    if title:
        entry.title = title
    if content is not None:
        entry.content = content
    if category is not None:
        entry.category = category
    if tags is not None:
        entry.tags = [t.strip() for t in tags.split(",")]
    
    entry.is_indexed = False
    
    db.commit()
    db.refresh(entry)
    
    return entry


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.MANAGE_KNOWLEDGE_BASE)
    
    entry = db.query(KnowledgeBaseEntry).filter(KnowledgeBaseEntry.id == entry_id).first()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base entry not found"
        )
    
    if entry.file_id:
        file_record = db.query(FileModel).filter(FileModel.id == entry.file_id).first()
        if file_record and os.path.exists(file_record.file_path):
            os.remove(file_record.file_path)
    
    db.delete(entry)
    db.commit()
