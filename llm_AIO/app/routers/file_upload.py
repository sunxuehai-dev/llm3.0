"""
通用文件上传路由
"""
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from convert_url import FileConverter
from app.database import get_convert_db as get_db
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter(tags=["file-upload"])

def get_converter(file_category: str, db: Session = Depends(get_db)):
    """获取文件转换器实例"""
    storage_type = os.getenv('STORAGE_TYPE', 'local')
    server_url = os.getenv('SERVER_URL', 'http://localhost:8000')
    
    config = {
        'server_url': server_url
    }
    
    # 如果使用 OSS 存储，添加 OSS 配置
    if storage_type == 'oss':
        config.update({
            'endpoint': os.getenv('OSS_ENDPOINT'),
            'access_key_id': os.getenv('OSS_ACCESS_KEY_ID'),
            'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET'),
            'bucket_name': os.getenv('OSS_BUCKET_NAME'),
            'region': os.getenv('OSS_REGION')
        })
    
    return FileConverter(
        session=db,
        file_category=file_category,
        storage_type=storage_type,
        **config
    )

@router.post("/api/{file_category}/upload")
async def upload_file(
    file_category: str = Path(..., pattern="^(video|audio|image)$"),
    file: UploadFile = File(...),
    user_id: str = Form(None),
    db: Session = Depends(get_db)
):
    """上传文件"""
    converter = get_converter(file_category, db)
    file_record, error = await converter.upload(file, user_id=user_id)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return {
        "success": True,
        f"{file_category}_id": file_record.id,
        "url": converter.get_url(file_record.id),
        "message": "上传成功"
    }

@router.get("/api/{file_category}/list")
async def list_files(
    file_category: str = Path(..., pattern="^(video|audio|image)$"),
    user_id: str = None,
    page: int = 1,
    per_page: int = 20,
    db: Session = Depends(get_db)
):
    """获取文件列表"""
    converter = get_converter(file_category, db)
    return converter.list(user_id=user_id, page=page, per_page=per_page)

@router.get("/api/{file_category}/{file_id}")
async def get_file(
    file_category: str = Path(..., pattern="^(video|audio|image)$"),
    file_id: str = Path(...),
    token: str = None,
    db: Session = Depends(get_db)
):
    """获取文件URL"""
    converter = get_converter(file_category, db)
    url = converter.get_url(file_id, token=token)
    if not url:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"url": url}

@router.delete("/api/{file_category}/{file_id}")
async def delete_file(
    file_category: str = Path(..., pattern="^(video|audio|image)$"),
    file_id: str = Path(...),
    db: Session = Depends(get_db)
):
    """删除文件"""
    converter = get_converter(file_category, db)
    success = converter.delete(file_id)
    if not success:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {"success": True, "message": "删除成功"}