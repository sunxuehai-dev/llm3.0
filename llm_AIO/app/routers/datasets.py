"""
训练集上传与管理接口
支持 LlamaFactory 格式（Alpaca/ShareGPT），存储到 OSS
"""
import logging

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query

from app.database import get_convert_db as get_db
from app.services import dataset_service
from app.models.user import User
from app.security import get_current_user
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/playground/datasets", tags=["datasets"])

# 数据类型、用途、格式枚举（前端可选）
DATA_TYPES = [
    "text_conversation",  # 文本对话
    "text2image",         # 文生图
    "image_classification",
    "text_classification",
    "pretrain",           # 预训练
    "dpo",                # 偏好训练
    "kto",
    "tool_call",
]
DATASET_TYPES = ["train", "inference"]
DATA_USAGES = ["sft", "dpo", "kto", "pretrain"]
DATA_FORMATS = ["alpaca", "sharegpt"]


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=255),
    description: str = Form(""),
    data_type: str = Form("text_conversation"),
    dataset_type: str = Form("train"),
    data_usage: str = Form("sft"),
    data_format: str = Form("alpaca"),
    user_id: str = Form(None),
    db: Session = Depends(get_db),
):
    """
    上传训练集
    - 格式要求符合 LlamaFactory：Alpaca 或 ShareGPT
    - 支持 json/jsonl/csv，最大 500MB
    - 数据存入已配置的 OSS（或本地）
    """
    if data_type not in DATA_TYPES:
        raise HTTPException(400, detail=f"data_type 必须在 {DATA_TYPES} 中")
    if dataset_type not in DATASET_TYPES:
        raise HTTPException(400, detail=f"dataset_type 必须在 {DATASET_TYPES} 中")
    if data_usage not in DATA_USAGES:
        raise HTTPException(400, detail=f"data_usage 必须在 {DATA_USAGES} 中")
    if data_format not in DATA_FORMATS:
        raise HTTPException(400, detail=f"data_format 必须在 {DATA_FORMATS} 中")

    try:
        result, err = await dataset_service.upload_dataset(
            db,
            file,
            name=name,
            description=description or None,
            data_type=data_type,
            dataset_type=dataset_type,
            data_usage=data_usage,
            data_format=data_format,
            user_id=user_id if user_id else None,
        )
    except Exception as e:
        logger.exception("dataset upload failed")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

    if err:
        raise HTTPException(status_code=400, detail=err)
    return {"success": True, "dataset": result, "message": "上传成功"}


@router.get("/list")
async def list_datasets(
    user_id: str = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """获取训练集列表"""
    return dataset_service.list_datasets(db, user_id=user_id, page=page, per_page=per_page)


@router.get("/options")
async def get_options():
    """获取前端选项：数据类型、用途、格式"""
    return {
        "data_types": DATA_TYPES,
        "dataset_types": DATASET_TYPES,
        "data_usages": DATA_USAGES,
        "data_formats": DATA_FORMATS,
    }


@router.delete("/{dataset_id}", response_model=dict)
async def delete_dataset(
    dataset_id: str,
    delete_file: bool = Query(False, description="是否同时删除本地训练集文件（仅 STORAGE_TYPE=local 时有效）"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    删除训练集：
    - 默认仅删除元数据记录（DatasetMetadata）
    - 若 delete_file=true 且 STORAGE_TYPE=local，则同时尝试删除 uploads/datasets 下对应的源文件
    """
    if not dataset_id:
        raise HTTPException(status_code=400, detail="缺少 dataset_id")

    # 先查出元数据，用于权限校验
    from convert_url import DatasetMetadata  # 延迟导入，避免循环

    meta = db.query(DatasetMetadata).filter_by(id=dataset_id).first()
    if not meta:
        raise HTTPException(status_code=404, detail="训练集不存在或已被删除")

    # 权限规则：
    # - super_admin / teacher：可删除任何训练集
    # - student：仅能删除自己上传的训练集（meta.user_id == 当前用户 id 字符串）
    if current_user.role == "student":
        if not meta.user_id or str(meta.user_id) != str(current_user.id):
            raise HTTPException(status_code=403, detail="无权限删除其他用户上传的训练集")

    ok = dataset_service.delete_dataset(db, dataset_id, delete_file=delete_file)
    if not ok:
        raise HTTPException(status_code=404, detail="训练集不存在或已被删除")
    return {"success": True, "message": "训练集已删除"}
