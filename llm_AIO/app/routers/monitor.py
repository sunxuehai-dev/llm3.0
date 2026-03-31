import psutil
import os
from app.datetime_utils import utc_now, serialize_datetime_for_api_response
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.database import get_db
from app.models.user import User, SystemMetrics
from app.schemas import SystemMetricsResponse, DashboardMetrics
from app.security import get_current_user
from app.permissions import require_permission, Permission

router = APIRouter(prefix="/monitor", tags=["System Monitor"])


def get_gpu_usage():
    """
    获取所有GPU的使用情况
    返回格式: {
        "gpu_count": int,
        "gpus": [
            {
                "gpu_id": int,
                "name": str,
                "utilization_percent": float,
                "memory_used_mb": float,
                "memory_total_mb": float,
                "memory_used_gb": float,
                "memory_total_gb": float,
                "memory_utilization_percent": float,
                "temperature": float (可选)
            }
        ],
        "total_memory_used_gb": float,
        "total_memory_total_gb": float,
        "average_utilization": float
    }
    """
    try:
        import subprocess
        # 获取GPU基本信息：索引、名称、利用率、显存使用、显存总量、温度
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return None
        
        gpus = []
        total_memory_used_mb = 0
        total_memory_total_mb = 0
        total_utilization = 0
        
        for line in lines:
            if not line.strip():
                continue
            # 使用更健壮的解析方式，处理可能的空格变化
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                try:
                    gpu_id = int(parts[0])
                    gpu_name = parts[1]
                    utilization = float(parts[2])
                    memory_used_mb = float(parts[3])
                    memory_total_mb = float(parts[4])
                    # 温度字段可能不存在或为空
                    temperature = None
                    if len(parts) > 5 and parts[5].strip():
                        try:
                            temperature = float(parts[5])
                        except (ValueError, IndexError):
                            temperature = None
                    
                    memory_used_gb = memory_used_mb / 1024.0
                    memory_total_gb = memory_total_mb / 1024.0
                    memory_utilization_percent = (memory_used_mb / memory_total_mb * 100) if memory_total_mb > 0 else 0
                    
                    gpu_info = {
                        "gpu_id": gpu_id,
                        "name": gpu_name,
                        "utilization_percent": round(utilization, 2),
                        "memory_used_mb": round(memory_used_mb, 2),
                        "memory_total_mb": round(memory_total_mb, 2),
                        "memory_used_gb": round(memory_used_gb, 2),
                        "memory_total_gb": round(memory_total_gb, 2),
                        "memory_utilization_percent": round(memory_utilization_percent, 2)
                    }
                    
                    if temperature is not None:
                        gpu_info["temperature"] = round(temperature, 1)
                    
                    gpus.append(gpu_info)
                    total_memory_used_mb += memory_used_mb
                    total_memory_total_mb += memory_total_mb
                    total_utilization += utilization
                except (ValueError, IndexError) as e:
                    # 跳过解析失败的行
                    continue
        
        if not gpus:
            return None
        
        return {
            "gpu_count": len(gpus),
            "gpus": gpus,
            "total_memory_used_gb": round(total_memory_used_mb / 1024.0, 2),
            "total_memory_total_gb": round(total_memory_total_mb / 1024.0, 2),
            "average_utilization": round(total_utilization / len(gpus), 2) if gpus else 0
        }
    except FileNotFoundError:
        # nvidia-smi 命令不存在
        return None
    except subprocess.TimeoutExpired:
        # 命令执行超时
        return None
    except Exception as e:
        # 其他异常，记录但不抛出
        return None


def get_storage_usage():
    disk = psutil.disk_usage('/')
    return {
        "used": disk.used / (1024**3),
        "total": disk.total / (1024**3)
    }


@router.get("/health", response_model=SystemMetricsResponse)
async def get_system_health(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    gpu_info = get_gpu_usage()
    storage_info = get_storage_usage()
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    online_teachers = db.query(User).filter(
        User.role == "teacher",
        User.is_active == True
    ).count()
    
    online_students = db.query(User).filter(
        User.role == "student",
        User.is_active == True
    ).count()
    
    return SystemMetricsResponse(
        gpu_usage=gpu_info["average_utilization"] if gpu_info else None,
        cpu_usage=cpu_usage,
        storage_used=round(storage_info["used"], 2),
        storage_total=round(storage_info["total"], 2),
        storage_unit="GB",
        cpu_unit="percent",
        gpu_unit="percent" if gpu_info else None,
        online_teachers=online_teachers,
        online_students=online_students,
        timestamp=utc_now()
    )


@router.get("/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    total_teachers = db.query(User).filter(User.role == "teacher").count()
    total_students = db.query(User).filter(User.role == "student").count()
    
    gpu_info = get_gpu_usage()
    storage_info = get_storage_usage()
    cpu_usage = psutil.cpu_percent(interval=0.1)
    
    system_health = SystemMetricsResponse(
        gpu_usage=gpu_info["average_utilization"] if gpu_info else None,
        cpu_usage=cpu_usage,
        storage_used=round(storage_info["used"], 2),
        storage_total=round(storage_info["total"], 2),
        storage_unit="GB",
        cpu_unit="percent",
        gpu_unit="percent" if gpu_info else None,
        online_teachers=db.query(User).filter(User.role == "teacher", User.is_active == True).count(),
        online_students=db.query(User).filter(User.role == "student", User.is_active == True).count(),
        timestamp=utc_now()
    )
    
    recent_users = db.query(User).order_by(User.created_at.desc()).limit(5).all()
    recent_activities = [
        {
            "type": "user_created",
            "description": f"New {user.role}: {user.username}",
            "timestamp": serialize_datetime_for_api_response(user.created_at) or "",
        }
        for user in recent_users
    ]
    
    return DashboardMetrics(
        total_teachers=total_teachers,
        total_students=total_students,
        total_resources=[],
        recent_activities=recent_activities,
        system_health=system_health
    )


@router.get("/gpu")
async def get_gpu_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取详细的GPU使用情况
    返回所有GPU的详细信息，包括每个GPU的利用率、显存使用等
    """
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    gpu_info = get_gpu_usage()
    
    if gpu_info:
        # 添加单位说明，方便前端理解
        gpu_info["units"] = {
            "utilization": "percent",
            "memory_mb": "MB",
            "memory_gb": "GB",
            "memory_utilization": "percent",
            "temperature": "Celsius"
        }
        return gpu_info
    
    return {
        "message": "GPU information not available",
        "gpu_count": 0,
        "gpus": [],
        "units": {
            "utilization": "percent",
            "memory_mb": "MB",
            "memory_gb": "GB",
            "memory_utilization": "percent",
            "temperature": "Celsius"
        }
    }


@router.get("/storage")
async def get_storage_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    storage_info = get_storage_usage()
    
    return {
        "storage_used": round(storage_info["used"], 2),
        "storage_total": round(storage_info["total"], 2),
        "storage_free": round(storage_info["total"] - storage_info["used"], 2),
        "storage_utilization_percent": round(
            (storage_info["used"] / storage_info["total"]) * 100, 2
        ),
        "unit": "GB"  # 明确标注单位
    }


@router.get("/memory")
async def get_memory_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    memory = psutil.virtual_memory()
    
    return {
        "memory_used": round(memory.used / (1024**3), 2),
        "memory_total": round(memory.total / (1024**3), 2),
        "memory_available": round(memory.available / (1024**3), 2),
        "memory_utilization_percent": memory.percent,
        "unit": "GB"  # 明确标注单位
    }


@router.get("/cpu")
async def get_cpu_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.VIEW_SYSTEM_METRICS)
    
    cpu_freq = psutil.cpu_freq()
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "cpu_frequency": cpu_freq.current if cpu_freq else None,
        "cpu_percent_unit": "percent",
        "cpu_frequency_unit": "MHz" if cpu_freq else None
    }
