"""
训练集上传与管理服务
支持 LlamaFactory 格式，存储到 OSS
"""
import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from convert_url import FileConverter, File, DatasetMetadata
from app.services.dataset_validator import (
    validate_dataset_format,
    make_dataset_info_name,
    FORMAT_ALPACA,
    FORMAT_SHAREGPT,
)

logger = logging.getLogger(__name__)


def get_dataset_converter(session, storage_type: str = None, server_url: str = None):
    """获取 dataset 类别的 FileConverter"""
    storage_type = storage_type or os.getenv("STORAGE_TYPE", "local")
    server_url = server_url or os.getenv("SERVER_URL", "http://localhost:8000")
    config = {"server_url": server_url}
    if storage_type == "oss":
        config.update({
            "endpoint": os.getenv("OSS_ENDPOINT"),
            "access_key_id": os.getenv("OSS_ACCESS_KEY_ID"),
            "access_key_secret": os.getenv("OSS_ACCESS_KEY_SECRET"),
            "bucket_name": os.getenv("OSS_BUCKET_NAME"),
            "region": os.getenv("OSS_REGION"),
        })
    return FileConverter(
        session=session,
        file_category="dataset",
        storage_type=storage_type,
        **config,
    )


async def upload_dataset(
    session,
    file,
    name: str,
    description: Optional[str],
    data_type: str,
    dataset_type: str,
    data_usage: str,
    data_format: str,
    user_id: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    上传训练集：只读一次 body -> 校验格式 -> 存本地/OSS -> 写入元数据。
    避免重复读取和全量解析导致大文件变慢。
    :return: (dataset_dict, error_message)
    """
    try:
        import io
        filename = getattr(file, "filename", None) or getattr(file, "name", "data.json")
        # 只读一次，后续校验和存储共用
        content = await file.read()

        # 格式校验（validator 内对 jsonl 已改为只解析前 100 条）
        ok, err, _ = validate_dataset_format(content, filename, data_format)
        if not ok:
            return None, err or "格式校验失败"

        # 用内存中的 content 交给 converter，避免 converter 再次 read(file) 导致重复读
        buf = io.BytesIO(content)
        buf.filename = filename
        converter = get_dataset_converter(session)
        file_record, upload_err = await converter.upload(buf, user_id=user_id)
        if upload_err:
            return None, upload_err

        # 写入元数据
        meta = DatasetMetadata(
            name=name.strip(),
            description=description or "",
            data_type=data_type,
            dataset_type=dataset_type,
            data_usage=data_usage,
            data_format=data_format,
            file_id=file_record.id,
            user_id=user_id,
        )
        session.add(meta)
        session.commit()

        info_name = make_dataset_info_name(meta.id, name)
        return {
            "id": meta.id,
            "name": meta.name,
            "description": meta.description,
            "data_type": meta.data_type,
            "dataset_type": meta.dataset_type,
            "data_usage": meta.data_usage,
            "data_format": meta.data_format,
            "file_id": meta.file_id,
            "dataset_info_name": info_name,
            "created_at": meta.created_at.isoformat() if meta.created_at else None,
        }, None
    except Exception as e:
        logger.exception("upload_dataset error")
        return None, f"上传失败: {str(e)}"


def prepare_dataset_for_training(
    session, dataset_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    为 LlamaFactory 训练准备数据集目录
    1. 下载/复制数据文件到本地缓存目录
    2. 生成 dataset_info.json
    :return: (dataset_dir, dataset_info_name) 或 (None, None)
    """
    meta = session.query(DatasetMetadata).filter_by(id=dataset_id).first()
    if not meta:
        return None, None
    file_rec = session.query(File).filter_by(id=meta.file_id).first()
    if not file_rec:
        return None, None

    info_name = make_dataset_info_name(meta.id, meta.name)
    cache_dir = Path("./uploads/datasets_cache") / dataset_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_file = cache_dir / file_rec.filename

    storage_type = os.getenv("STORAGE_TYPE", "local")
    if storage_type == "oss":
        try:
            import oss2
            endpoint = os.getenv("OSS_ENDPOINT", "").replace("https://", "").replace("http://", "")
            auth = oss2.Auth(os.getenv("OSS_ACCESS_KEY_ID"), os.getenv("OSS_ACCESS_KEY_SECRET"))
            bucket = oss2.Bucket(auth, endpoint, os.getenv("OSS_BUCKET_NAME"))
            bucket.get_object_to_file(file_rec.file_path, str(target_file))
        except Exception:
            return None, None
    else:
        base_path = Path("./uploads/datasets")
        src = base_path / file_rec.file_path
        if src.exists():
            shutil.copy2(src, target_file)
        else:
            return None, None

    # 写入 dataset_info.json
    entry = {"file_name": file_rec.filename}
    if meta.data_format == FORMAT_SHAREGPT:
        entry["formatting"] = "sharegpt"
        entry["columns"] = {"messages": "messages"}
        entry["tags"] = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        }
    else:
        entry["formatting"] = "alpaca"

    info_path = cache_dir / "dataset_info.json"
    info_data = {info_name: entry}
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
            info_data[info_name] = entry
        except Exception:
            pass
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)

    return str(cache_dir), info_name




def list_datasets(
    session,
    user_id: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
) -> Dict:
    """列出训练集"""
    q = session.query(DatasetMetadata)
    if user_id:
        q = q.filter_by(user_id=user_id)
    total = q.count()
    offset = (page - 1) * per_page
    items = q.order_by(DatasetMetadata.created_at.desc()).offset(offset).limit(per_page).all()
    pages = (total + per_page - 1) // per_page if total else 0
    datasets = []
    for m in items:
        info_name = make_dataset_info_name(m.id, m.name)
        datasets.append({
            **m.to_dict(),
            "dataset_info_name": info_name,
        })
    return {
        "datasets": datasets,
        "pagination": {"page": page, "per_page": per_page, "total": total, "pages": pages},
    }


def delete_dataset(session, dataset_id: str, delete_file: bool = False) -> bool:
    """
    删除训练集元数据；可选地删除本地存储的原始文件。
    - 若 STORAGE_TYPE=oss，则仅删除元数据，不会删除 OSS 上的实际文件。
    :return: True 表示删除成功（找到并删除），False 表示未找到
    """
    if not dataset_id:
        return False

    meta = session.query(DatasetMetadata).filter_by(id=dataset_id).first()
    if not meta:
        return False

    # 可选删除本地文件（仅在 STORAGE_TYPE=local 时生效）
    if delete_file:
        storage_type = os.getenv("STORAGE_TYPE", "local")
        if storage_type != "oss" and meta.file_id:
            file_rec = session.query(File).filter_by(id=meta.file_id).first()
            if file_rec and file_rec.file_path:
                base_path = Path("./uploads/datasets")
                src = base_path / file_rec.file_path
                try:
                    if src.exists():
                        src.unlink()
                except OSError:
                    pass

    # 删除元数据记录
    session.delete(meta)
    session.commit()
    return True
