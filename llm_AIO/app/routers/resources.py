from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import psutil
import subprocess

from app.database import get_db
from app.models.user import User, SystemResource, ResourceAllocation
from app.schemas import ResourceCreate, ResourceResponse, ResourceAllocationCreate, ResourceAllocationResponse, UserResponse
from app.security import get_current_user
from app.permissions import require_permission, Permission

router = APIRouter(prefix="/resources", tags=["Resource Management"])


def sync_resources_from_hardware(db: Session) -> List[SystemResource]:
    """
    从实际硬件同步资源信息到数据库
    返回更新后的资源列表
    """
    resources = []
    
    # 获取CPU核心数
    try:
        cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 16
        cpu_resource = db.query(SystemResource).filter(
            SystemResource.name == "CPU Cores"
        ).first()
        
        if cpu_resource:
            cpu_resource.total_amount = float(cpu_cores)
            resources.append(cpu_resource)
        else:
            cpu_resource = SystemResource(
                name="CPU Cores",
                resource_type="compute",
                total_amount=float(cpu_cores),
                unit="cores"
            )
            db.add(cpu_resource)
            resources.append(cpu_resource)
    except Exception:
        cpu_resource = db.query(SystemResource).filter(
            SystemResource.name == "CPU Cores"
        ).first()
        if cpu_resource:
            resources.append(cpu_resource)
    
    # 获取GPU显存
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            total_memory_mb = 0
            for line in lines:
                if line.strip():
                    try:
                        memory_mb = float(line.strip())
                        total_memory_mb += memory_mb
                    except ValueError:
                        continue
            
            if total_memory_mb > 0:
                gpu_memory_gb = round(total_memory_mb / 1024.0, 2)
                gpu_resource = db.query(SystemResource).filter(
                    SystemResource.name == "GPU Memory"
                ).first()
                
                if gpu_resource:
                    gpu_resource.total_amount = gpu_memory_gb
                    resources.append(gpu_resource)
                else:
                    gpu_resource = SystemResource(
                        name="GPU Memory",
                        resource_type="compute",
                        total_amount=gpu_memory_gb,
                        unit="GB"
                    )
                    db.add(gpu_resource)
                    resources.append(gpu_resource)
            else:
                gpu_resource = db.query(SystemResource).filter(
                    SystemResource.name == "GPU Memory"
                ).first()
                if gpu_resource:
                    resources.append(gpu_resource)
        else:
            gpu_resource = db.query(SystemResource).filter(
                SystemResource.name == "GPU Memory"
            ).first()
            if gpu_resource:
                resources.append(gpu_resource)
    except Exception:
        gpu_resource = db.query(SystemResource).filter(
            SystemResource.name == "GPU Memory"
        ).first()
        if gpu_resource:
            resources.append(gpu_resource)
    
    # 获取存储容量
    try:
        disk = psutil.disk_usage('/')
        storage_total_gb = round(disk.total / (1024**3), 2)
        storage_resource = db.query(SystemResource).filter(
            SystemResource.name == "Storage"
        ).first()
        
        if storage_resource:
            storage_resource.total_amount = storage_total_gb
            resources.append(storage_resource)
        else:
            storage_resource = SystemResource(
                name="Storage",
                resource_type="storage",
                total_amount=storage_total_gb,
                unit="GB"
            )
            db.add(storage_resource)
            resources.append(storage_resource)
    except Exception:
        storage_resource = db.query(SystemResource).filter(
            SystemResource.name == "Storage"
        ).first()
        if storage_resource:
            resources.append(storage_resource)
    
    db.commit()
    
    # 刷新所有更新的资源
    for resource in resources:
        db.refresh(resource)
    
    return resources


@router.get("/", response_model=List[ResourceResponse])
async def get_resources(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取所有资源列表
    自动从实际硬件同步资源信息，而不是返回数据库中的假数据
    """
    require_permission(current_user.role, Permission.ALLOCATE_RESOURCES)
    
    # 从实际硬件获取资源信息，而不是使用数据库中的假数据
    return sync_resources_from_hardware(db)


@router.get("/available", response_model=List[ResourceResponse])
async def get_available_resources(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取可用资源列表（所有用户可访问）
    自动从实际硬件同步资源信息
    """
    # 从实际硬件获取资源信息，而不是使用数据库中的假数据
    return sync_resources_from_hardware(db)


@router.post("/", response_model=ResourceResponse, status_code=status.HTTP_201_CREATED)
async def create_resource(
    resource_data: ResourceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.ALLOCATE_RESOURCES)
    
    existing = db.query(SystemResource).filter(
        SystemResource.name == resource_data.name
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resource with this name already exists"
        )
    
    resource = SystemResource(
        name=resource_data.name,
        resource_type=resource_data.resource_type,
        total_amount=resource_data.total_amount,
        unit=resource_data.unit
    )
    
    db.add(resource)
    db.commit()
    db.refresh(resource)
    
    return resource


@router.get("/user/{user_id}", response_model=List[ResourceAllocationResponse])
async def get_user_resources(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role == "student" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot view other users' resources"
        )
    
    allocations = db.query(ResourceAllocation).filter(
        ResourceAllocation.user_id == user_id
    ).all()
    
    return allocations


@router.post("/allocate", response_model=ResourceAllocationResponse, status_code=status.HTTP_201_CREATED)
async def allocate_resource(
    allocation_data: ResourceAllocationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.ALLOCATE_RESOURCES)
    
    resource = db.query(SystemResource).filter(
        SystemResource.id == allocation_data.resource_id
    ).first()
    
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
    
    user = db.query(User).filter(User.id == allocation_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    existing_allocations = db.query(ResourceAllocation).filter(
        ResourceAllocation.user_id == allocation_data.user_id,
        ResourceAllocation.resource_id == allocation_data.resource_id
    ).all()
    
    total_allocated = sum(alloc.allocated_amount for alloc in existing_allocations)
    
    if total_allocated + allocation_data.allocated_amount > resource.total_amount:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient resource. Available: {resource.total_amount - total_allocated} {resource.unit}"
        )
    
    allocation = ResourceAllocation(
        user_id=allocation_data.user_id,
        resource_id=allocation_data.resource_id,
        allocated_amount=allocation_data.allocated_amount,
        expires_at=allocation_data.expires_at,
        notes=allocation_data.notes
    )
    
    db.add(allocation)
    db.commit()
    db.refresh(allocation)
    
    return allocation


@router.delete("/{allocation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_resource(
    allocation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    require_permission(current_user.role, Permission.ALLOCATE_RESOURCES)
    
    allocation = db.query(ResourceAllocation).filter(
        ResourceAllocation.id == allocation_id
    ).first()
    
    if not allocation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Allocation not found"
        )
    
    db.delete(allocation)
    db.commit()


@router.post("/sync-from-hardware", response_model=List[ResourceResponse])
async def sync_resources_from_hardware_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    从实际硬件同步资源信息到数据库（手动触发）
    自动检测CPU核心数、GPU显存、存储容量，并更新或创建对应的资源记录
    """
    require_permission(current_user.role, Permission.ALLOCATE_RESOURCES)
    
    return sync_resources_from_hardware(db)
