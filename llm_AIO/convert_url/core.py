"""
核心功能 - 文件上传与URL转换
"""
import os
import re
import secrets
import inspect
from datetime import datetime
from pathlib import Path
from sqlalchemy import Column, String, Integer, Boolean, DateTime
from sqlalchemy.orm import Session
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


def secure_filename(filename):
    """安全的文件名处理"""
    if not filename:
        return ""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename


class File(Base):
    """通用文件模型"""
    __tablename__ = 'files'
    
    id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    file_category = Column(String(20), nullable=False, index=True)  # video, audio, image, dataset
    user_id = Column(String(64), nullable=True, index=True)
    access_token = Column(String(128), nullable=True, unique=True, index=True)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='active')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.id:
            self.id = secrets.token_urlsafe(32)
        if not self.access_token:
            self.access_token = secrets.token_urlsafe(48)
    
    def to_dict(self, include_token=False):
        data = {
            'id': self.id, 'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size, 'file_type': self.file_type,
            'file_category': self.file_category,
            'user_id': self.user_id, 'is_public': self.is_public,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'status': self.status
        }
        if include_token:
            data['access_token'] = self.access_token
        return data


class DatasetMetadata(Base):
    """训练集元数据模型（LlamaFactory 兼容）"""
    __tablename__ = 'dataset_metadata'

    id = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False, index=True)  # 数据集显示名称
    description = Column(String(2000), nullable=True)
    data_type = Column(String(64), nullable=False)  # text_conversation, text2image, image_classification, text_classification, etc.
    dataset_type = Column(String(32), nullable=False, default='train')  # train, inference
    data_usage = Column(String(64), nullable=False, default='sft')  # sft, dpo, kto, pretrain
    data_format = Column(String(32), nullable=False)  # alpaca, sharegpt
    file_id = Column(String(64), nullable=False, index=True)  # 关联 File 表
    user_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.id:
            self.id = secrets.token_urlsafe(32)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'data_type': self.data_type,
            'dataset_type': self.dataset_type,
            'data_usage': self.data_usage,
            'data_format': self.data_format,
            'file_id': self.file_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TrainingJob(Base):
    """LLMFactory 训练任务记录（用于查看已完成训练及成功/失败状态）"""
    __tablename__ = 'training_jobs'

    id = Column(String(64), primary_key=True)
    output_dir = Column(String(1024), nullable=False, index=True)
    task_type = Column(String(32), nullable=False)  # lora, qlora, full
    status = Column(String(32), nullable=False, index=True)  # running, success, failed
    error_message = Column(String(4096), nullable=True)
    dataset_id = Column(String(64), nullable=True, index=True)  # 使用的训练集 ID（若有）
    data_type = Column(String(64), nullable=True)  # 训练集类型，如 text_conversation, text2image
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.id:
            self.id = secrets.token_urlsafe(32)

    def to_dict(self):
        return {
            'id': self.id,
            'output_dir': self.output_dir,
            'task_type': self.task_type,
            'status': self.status,
            'error_message': self.error_message,
            'dataset_id': self.dataset_id,
            'data_type': self.data_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class FileConverter:
    """通用文件转换器"""
    
    # 文件类型配置
    FILE_CONFIGS = {
        'video': {
            'extensions': {'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv', 'webm', 'm4v'},
            'max_size': 1024 * 1024 * 1024,  # 1GB
            'base_path': './uploads/videos',
            'url_prefix': '/api/video',
            'content_type': 'video/mp4'
        },
        'audio': {
            'extensions': {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac', 'wma', 'opus'},
            'max_size': 100 * 1024 * 1024,  # 100MB
            'base_path': './uploads/audios',
            'url_prefix': '/api/audio',
            'content_type': 'audio/mpeg'
        },
        'image': {
            'extensions': {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg', 'ico'},
            'max_size': 10 * 1024 * 1024,  # 10MB
            'base_path': './uploads/images',
            'url_prefix': '/api/image',
            'content_type': 'image/jpeg'
        },
        'dataset': {
            'extensions': {'json', 'jsonl', 'csv'},
            'max_size': 500 * 1024 * 1024,  # 500MB
            'base_path': './uploads/datasets',
            'url_prefix': '/api/dataset',
            'content_type': 'application/json'
        }
    }
    
    def __init__(self, session: Session, file_category='video', storage_type='local', **storage_config):
        """
        初始化转换器
        
        Args:
            session: SQLAlchemy Session 实例
            file_category: 文件类别 'video', 'audio', 'image'
            storage_type: 存储类型 'local' 或 'oss'
            **storage_config: 存储配置（会覆盖默认配置）
        """
        if file_category not in self.FILE_CONFIGS:
            raise ValueError(f"不支持的文件类别: {file_category}")
        
        self.session = session
        self.file_category = file_category
        self.storage_type = storage_type
        self.config = self.FILE_CONFIGS[file_category].copy()
        self.config.update(storage_config)
    
    def _get_storage(self):
        """获取存储实例"""
        if self.storage_type == 'oss':
            return self._get_oss_storage()
        return self._get_local_storage()
    
    def _get_local_storage(self):
        """本地存储"""
        base_path = Path(self.config.get('base_path'))
        base_path.mkdir(parents=True, exist_ok=True)
        config = self.config
        
        class LocalStorage:
            def __init__(self):
                self.base_path = base_path
                self.config = config
            
            async def save_file(self, file, filename):
                date_dir = datetime.now().strftime('%Y/%m/%d')
                target_dir = self.base_path / date_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                file_path = target_dir / filename
                
                # 检测是否为 FastAPI UploadFile
                is_upload_file = False
                if hasattr(file, 'read'):
                    try:
                        is_upload_file = inspect.iscoroutinefunction(file.read)
                    except (TypeError, AttributeError):
                        is_upload_file = (
                            hasattr(file, 'spool_max_size') or 
                            type(file).__name__ == 'UploadFile'
                        )
                else:
                    is_upload_file = (
                        hasattr(file, 'spool_max_size') or 
                        type(file).__name__ == 'UploadFile'
                    )
                
                with open(str(file_path), 'wb') as f:
                    if isinstance(file, bytes):
                        f.write(file)
                    elif hasattr(file, 'read'):
                        # 处理 FastAPI UploadFile (异步) 或其他文件对象
                        if is_upload_file:
                            # FastAPI UploadFile - 异步读取
                            content = await file.read()
                        else:
                            # 普通文件对象 - 同步读取
                            content = file.read()
                        
                        if isinstance(content, bytes):
                            f.write(content)
                        elif isinstance(content, str):
                            f.write(content.encode())
                        else:
                            f.write(content)
                    else:
                        f.write(file)
                
                return f"{date_dir}/{filename}"
            
            def get_file_url(self, file_path, file_id=None, token=None):
                server_url = self.config.get('server_url', 'http://localhost:8000')
                url_prefix = self.config.get('url_prefix')
                if self.config.get('enable_token_auth') and token:
                    return f"{server_url}{url_prefix}/{file_id}?token={token}"
                return f"{server_url}{url_prefix}/{file_id}" if file_id else f"{server_url}/uploads/{file_path}"
            
            def delete_file(self, file_path):
                full_path = self.base_path / file_path
                if full_path.exists():
                    full_path.unlink()
                    return True
                return False
        
        return LocalStorage()
    
    def _get_oss_storage(self):
        """OSS存储"""
        # 优先使用阿里云官方的 oss2 SDK，兼容性更好
        try:
            import oss2
            use_oss2 = True
        except ImportError:
            use_oss2 = False
        
        if use_oss2:
            # 使用 oss2 SDK（推荐）
            endpoint = self.config.get('endpoint')
            access_key_id = self.config.get('access_key_id')
            access_key_secret = self.config.get('access_key_secret')
            bucket_name = self.config.get('bucket_name')
            
            # 检查必要的配置
            if not endpoint or not access_key_id or not access_key_secret or not bucket_name:
                raise ValueError("OSS 配置不完整，请检查 .env 文件中的 OSS_* 配置项")
            
            # 移除 https:// 前缀（oss2 需要）
            if endpoint and isinstance(endpoint, str):
                if endpoint.startswith('https://'):
                    endpoint = endpoint[8:]
                elif endpoint.startswith('http://'):
                    endpoint = endpoint[7:]
            
            try:
                auth = oss2.Auth(access_key_id, access_key_secret)
                bucket = oss2.Bucket(auth, endpoint, bucket_name)
            except Exception as e:
                raise ValueError(f"OSS 初始化失败: {str(e)}，请检查 OSS 配置")
            content_type = self.config.get('content_type')
            config = self.config
            
            class OSSStorage:
                def __init__(self):
                    self.bucket = bucket
                    self.config = config
                    self.content_type = content_type
                
                async def save_file(self, file, filename):
                    date_dir = datetime.now().strftime('%Y/%m/%d')
                    file_category = self.config.get('file_category', 'files')
                    object_key = f"{file_category}s/{date_dir}/{filename}"
                    
                    # 检测是否为 FastAPI UploadFile
                    is_upload_file = False
                    if hasattr(file, 'read'):
                        try:
                            is_upload_file = inspect.iscoroutinefunction(file.read)
                        except (TypeError, AttributeError):
                            is_upload_file = (
                                hasattr(file, 'spool_max_size') or 
                                type(file).__name__ == 'UploadFile'
                            )
                    else:
                        is_upload_file = (
                            hasattr(file, 'spool_max_size') or 
                            type(file).__name__ == 'UploadFile'
                        )
                    
                    # 处理 FastAPI UploadFile (异步) 或其他文件对象
                    if is_upload_file:
                        # FastAPI UploadFile - 异步方法
                        if hasattr(file, 'seek'):
                            await file.seek(0)
                        content = await file.read()
                    elif hasattr(file, 'seek'):
                        # 普通文件对象，支持 seek
                        file.seek(0)
                        if hasattr(file, 'read'):
                            content = file.read()
                        else:
                            content = file
                    elif hasattr(file, 'read'):
                        # 普通文件对象，只支持 read
                        content = file.read()
                    else:
                        content = file
                    
                    if isinstance(content, str):
                        content = content.encode()
                    
                    # 确保 content 是 bytes
                    if not isinstance(content, bytes):
                        import io
                        if hasattr(content, 'read'):
                            content = content.read()
                        else:
                            content = bytes(content)
                    
                    # 使用 oss2 上传
                    self.bucket.put_object(
                        object_key,
                        content,
                        headers={'Content-Type': self.content_type}
                    )
                    return object_key
                
                def get_file_url(self, file_path, file_id=None, token=None, expire_hours=24):
                    # oss2 生成预签名 URL
                    url = self.bucket.sign_url('GET', file_path, expire_hours * 3600)
                    # Fix: 阿里云 OSS 预签名 URL 会对 object key 中的 '/' 进行编码 (%2F)，导致部分服务无法识别
                    # 我们手动替换回来
                    return url.replace('%2F', '/')
                
                def delete_file(self, file_path):
                    try:
                        self.bucket.delete_object(file_path)
                        return True
                    except:
                        return False
            
            return OSSStorage()
        else:
            # 回退到 boto3（如果 oss2 不可用）
            try:
                import boto3
                from botocore.config import Config as BotoConfig
                from datetime import timedelta
                
                # 配置 boto3 使用虚拟主机样式访问 OSS（阿里云 OSS 要求）
                boto_config = BotoConfig(
                    signature_version='s3v4',
                    region_name=self.config.get('region'),
                    s3={
                        'addressing_style': 'virtual',
                        'payload_signing_enabled': False  # 禁用 payload signing
                    }
                )
                
                client = boto3.client(
                    's3',
                    endpoint_url=self.config.get('endpoint'),
                    aws_access_key_id=self.config.get('access_key_id'),
                    aws_secret_access_key=self.config.get('access_key_secret'),
                    config=boto_config
                )
                bucket = self.config.get('bucket_name')
                content_type = self.config.get('content_type')
                config = self.config
                
                class OSSStorage:
                    def __init__(self):
                        self.client = client
                        self.bucket = bucket
                        self.config = config
                        self.content_type = content_type
                    
                    async def save_file(self, file, filename):
                        date_dir = datetime.now().strftime('%Y/%m/%d')
                        file_category = self.config.get('file_category', 'files')
                        object_key = f"{file_category}s/{date_dir}/{filename}"
                        
                        # 检测是否为 FastAPI UploadFile
                        is_upload_file = False
                        if hasattr(file, 'read'):
                            try:
                                is_upload_file = inspect.iscoroutinefunction(file.read)
                            except (TypeError, AttributeError):
                                is_upload_file = (
                                    hasattr(file, 'spool_max_size') or 
                                    type(file).__name__ == 'UploadFile'
                                )
                        else:
                            is_upload_file = (
                                hasattr(file, 'spool_max_size') or 
                                type(file).__name__ == 'UploadFile'
                            )
                        
                        # 处理 FastAPI UploadFile (异步) 或其他文件对象
                        if is_upload_file:
                            # FastAPI UploadFile - 异步方法
                            if hasattr(file, 'seek'):
                                await file.seek(0)
                            content = await file.read()
                        elif hasattr(file, 'seek'):
                            # 普通文件对象，支持 seek
                            file.seek(0)
                            if hasattr(file, 'read'):
                                content = file.read()
                            else:
                                content = file
                        elif hasattr(file, 'read'):
                            # 普通文件对象，只支持 read
                            content = file.read()
                        else:
                            content = file
                        
                        if isinstance(content, str):
                            content = content.encode()
                        
                        # 确保 content 是 bytes
                        if not isinstance(content, bytes):
                            import io
                            if hasattr(content, 'read'):
                                content = content.read()
                            else:
                                content = bytes(content)
                        
                        # 使用 put_object 而不是 upload_fileobj，避免 chunked encoding 问题
                        self.client.put_object(
                            Bucket=self.bucket,
                            Key=object_key,
                            Body=content,
                            ContentType=self.content_type
                        )
                        return object_key
                    
                    def get_file_url(self, file_path, file_id=None, token=None, expire_hours=24):
                        return self.client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': self.bucket, 'Key': file_path},
                            ExpiresIn=int(timedelta(hours=expire_hours).total_seconds())
                        )
                    
                    def delete_file(self, file_path):
                        try:
                            self.client.delete_object(Bucket=self.bucket, Key=file_path)
                            return True
                        except:
                            return False
                
                return OSSStorage()
            except ImportError:
                raise ImportError("使用OSS存储需要安装 oss2 (推荐) 或 boto3: pip install oss2 或 pip install boto3")
    
    async def upload(self, file, user_id=None, is_public=None):
        """上传文件"""
        filename = None
        if hasattr(file, 'filename'):
            filename = file.filename
        elif hasattr(file, 'name'):
            filename = file.name
        
        if not filename:
            return None, "未提供文件"
        
        # 验证文件
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        allowed = self.config.get('allowed_extensions', self.FILE_CONFIGS[self.file_category]['extensions'])
        if ext not in allowed:
            return None, f"不支持的文件格式: {ext}"
        
        # 检查大小
        file_size = 0
        # 检测是否为 FastAPI UploadFile（通过检查 read 方法是否为协程函数）
        is_upload_file = False
        if hasattr(file, 'read'):
            try:
                is_upload_file = inspect.iscoroutinefunction(file.read)
            except (TypeError, AttributeError):
                # 备用检测方法
                is_upload_file = (
                    hasattr(file, 'spool_max_size') or 
                    type(file).__name__ == 'UploadFile'
                )
        else:
            is_upload_file = (
                hasattr(file, 'spool_max_size') or 
                type(file).__name__ == 'UploadFile'
            )
        
        if is_upload_file:
            # FastAPI UploadFile - 使用 size 属性（如果可用）或异步读取
            if hasattr(file, 'size') and file.size is not None:
                file_size = file.size
            else:
                # 异步读取来获取大小
                try:
                    content = await file.read()
                    if isinstance(content, bytes):
                        file_size = len(content)
                    elif isinstance(content, str):
                        file_size = len(content.encode())
                    else:
                        file_size = 0
                    # 重置文件指针以便后续使用
                    if hasattr(file, 'seek'):
                        await file.seek(0)
                except Exception as e:
                    # 如果读取失败，尝试其他方法
                    file_size = 0
        elif hasattr(file, 'seek') and hasattr(file, 'tell'):
            # 普通文件对象，支持 seek 和 tell
            try:
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
            except (AttributeError, OSError):
                file_size = 0
        elif hasattr(file, 'read'):
            # 普通文件对象，只支持 read（同步）
            try:
                content = file.read()
                if isinstance(content, bytes):
                    file_size = len(content)
                elif isinstance(content, str):
                    file_size = len(content.encode())
                elif hasattr(content, '__len__'):
                    file_size = len(content)
                if hasattr(file, 'seek'):
                    file.seek(0)
            except Exception:
                file_size = 0
        elif isinstance(file, bytes):
            file_size = len(file)
        
        max_size = self.config.get('max_content_length', self.FILE_CONFIGS[self.file_category]['max_size'])
        if file_size > max_size:
            return None, f"文件大小超过限制: {max_size / 1024 / 1024:.0f}MB"
        
        try:
            storage = self._get_storage()
            original_filename = secure_filename(filename)
            unique_filename = f"{int(datetime.now().timestamp() * 1000)}_{secrets.token_hex(8)}.{ext}"
            file_path = await storage.save_file(file, unique_filename)
            
            file_record = File(
                filename=unique_filename,
                original_filename=original_filename,
                file_path=file_path,
                file_size=file_size,
                file_type=ext,
                file_category=self.file_category,
                user_id=user_id,
                is_public=is_public if is_public is not None else not self.config.get('enable_token_auth', False)
            )
            
            self.session.add(file_record)
            self.session.commit()
            
            return file_record, None
        except Exception as e:
            self.session.rollback()
            return None, f"上传失败: {str(e)}"
    
    def get_url(self, file_id, token=None):
        """获取文件URL"""
        file_record = self.session.query(File).filter_by(
            id=file_id, 
            file_category=self.file_category,
            status='active'
        ).first()
        if not file_record:
            return None
        
        storage = self._get_storage()
        url_token = token or (file_record.access_token if not file_record.is_public and self.config.get('enable_token_auth') else None)
        return storage.get_file_url(file_record.file_path, file_id=file_record.id, token=url_token)
    
    def delete(self, file_id):
        """删除文件"""
        file_record = self.session.query(File).filter_by(
            id=file_id,
            file_category=self.file_category,
            status='active'
        ).first()
        if not file_record:
            return False
        
        storage = self._get_storage()
        storage.delete_file(file_record.file_path)
        file_record.status = 'deleted'
        self.session.commit()
        return True
    
    def list(self, user_id=None, page=1, per_page=20):
        """获取文件列表"""
        query = self.session.query(File).filter_by(
            file_category=self.file_category,
            status='active'
        )
        if user_id:
            query = query.filter_by(user_id=user_id)
        
        total = query.count()
        offset = (page - 1) * per_page
        files_list = query.order_by(File.created_at.desc()).offset(offset).limit(per_page).all()
        
        storage = self._get_storage()
        files = []
        for file_record in files_list:
            data = file_record.to_dict(include_token=False)
            token = file_record.access_token if self.config.get('enable_token_auth') and not file_record.is_public else None
            data['url'] = storage.get_file_url(file_record.file_path, file_id=file_record.id, token=token)
            files.append(data)
        
        pages = (total + per_page - 1) // per_page if total > 0 else 0
        
        return {
            f'{self.file_category}s': files,
            'pagination': {'page': page, 'per_page': per_page, 'total': total, 'pages': pages}
        }