"""
数据库初始化脚本
创建数据库表、默认管理员用户和系统资源
"""
from app.database import user_engine, UserBase, UserSessionLocal
from app.models.user import User, SystemResource, RagVectorKbControl  # noqa: F401 — RagVectorKbControl 注册表结构
# 确保竞赛 ORM 也被加载到 metadata（从而在 create_all 时创建表）
from app.models.competition import (
    Competition,
    CompetitionEnrollment,
    Team,
    TeamMember,
    Submission,
    Review,
)
# 确保考试模块 ORM 被加载到 metadata
from app.models.exam import (
    QuestionBankItem,
    Exam,
    ExamQuestion,
    ExamAttempt,
    ExamAnswer,
)
from app.security import get_password_hash
import psutil
import subprocess


def get_actual_gpu_memory():
    """从实际硬件获取GPU显存总量（GB）"""
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
                return round(total_memory_mb / 1024.0, 2)  # 转换为GB
    except Exception:
        pass
    return None


def get_actual_cpu_cores():
    """从实际硬件获取CPU核心数"""
    try:
        return psutil.cpu_count(logical=False) or psutil.cpu_count() or 16
    except Exception:
        return 16


def get_actual_storage():
    """从实际硬件获取存储总量（GB）"""
    try:
        disk = psutil.disk_usage('/')
        return round(disk.total / (1024**3), 2)
    except Exception:
        return 1000.0


def init_db():
    """初始化数据库：创建表、默认管理员和系统资源"""
    # 首先创建所有数据库表
    UserBase.metadata.create_all(bind=user_engine)
    
    db = UserSessionLocal()
    try:
        # 检查是否已存在超级管理员
        if not db.query(User).filter(User.role == "super_admin").first():
            # 创建默认超级管理员
            super_admin = User(
                username="admin",
                email="admin@system.edu",
                full_name="System Administrator",
                hashed_password=get_password_hash("admin123"),
                role="super_admin",
                is_active=True
            )
            db.add(super_admin)
            
            # 从实际硬件获取资源信息
            cpu_cores = get_actual_cpu_cores()
            gpu_memory = get_actual_gpu_memory()
            storage_total = get_actual_storage()
            
            # 创建CPU资源
            cpu_resource = SystemResource(
                name="CPU Cores",
                resource_type="compute",
                total_amount=float(cpu_cores),
                unit="cores"
            )
            
            # 创建GPU资源（如果检测到GPU，使用实际值；否则使用默认值）
            if gpu_memory:
                gpu_resource = SystemResource(
                    name="GPU Memory",
                    resource_type="compute",
                    total_amount=gpu_memory,
                    unit="GB"
                )
                print(f"✓ Detected GPU memory: {gpu_memory} GB")
            else:
                gpu_resource = SystemResource(
                    name="GPU Memory",
                    resource_type="compute",
                    total_amount=24.0,
                    unit="GB"
                )
                print("⚠ GPU not detected, using default: 24.0 GB")
            
            # 创建存储资源
            storage_resource = SystemResource(
                name="Storage",
                resource_type="storage",
                total_amount=storage_total,
                unit="GB"
            )
            
            db.add(cpu_resource)
            db.add(gpu_resource)
            db.add(storage_resource)
            
            db.commit()
            print("\n" + "="*50)
            print("✓ Initial database setup completed!")
            print("="*50)
            print("\nDefault super admin created:")
            print("  Username: admin")
            print("  Password: admin123")
            print("  ⚠️  Please change the password after first login!")
            print(f"\nResources initialized:")
            print(f"  CPU Cores: {cpu_cores}")
            print(f"  GPU Memory: {gpu_resource.total_amount} GB")
            print(f"  Storage: {storage_total} GB")
            print("="*50 + "\n")
        else:
            print("✓ Database already initialized (super admin exists)")
    except Exception as e:
        db.rollback()
        print(f"✗ Error initializing database: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
