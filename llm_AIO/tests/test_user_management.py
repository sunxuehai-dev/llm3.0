"""
用户管理服务测试
测试认证、用户管理、资源管理等接口
"""
import requests
import json
import sys
import os

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://192.168.3.238:8000/api/v1"

# 测试结果输出目录
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 测试用的 Token（登录后获取）
access_token = None

def print_result(test_name, success, message="", data=None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data and not isinstance(data, dict):
        print(f"      响应: {data}")
    elif data:
        # 隐藏敏感信息
        safe_data = data.copy()
        if 'access_token' in safe_data:
            safe_data['access_token'] = safe_data['access_token'][:20] + "..."
        if 'hashed_password' in safe_data:
            safe_data['hashed_password'] = "***"
        print(f"      响应: {json.dumps(safe_data, ensure_ascii=False, indent=2)}")
    print()

def get_headers():
    """获取请求头（包含 Token）"""
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers

def test_health_check():
    """测试健康检查"""
    test_name = "健康检查"
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print_result(test_name, True, "服务正常", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}")
            return False
    except Exception as e:
        print_result(test_name, False, f"无法连接到服务器: {str(e)}")
        return False

def test_login():
    """测试用户登录"""
    global access_token
    test_name = "用户登录"
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={
                "username": "admin",
                "password": "admin123"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            access_token = result.get("access_token")
            print_result(test_name, True, f"用户: {result.get('full_name')}, 角色: {result.get('role')}", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_current_user():
    """测试获取当前用户信息"""
    test_name = "获取当前用户信息"
    
    try:
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result(test_name, True, f"用户名: {result.get('username')}", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_users():
    """测试获取用户列表"""
    test_name = "获取用户列表"
    
    try:
        response = requests.get(
            f"{BASE_URL}/users/",
            headers=get_headers(),
            params={"skip": 0, "limit": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            user_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"用户数量: {user_count}", result[:2] if isinstance(result, list) else result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_resources():
    """测试获取资源列表"""
    test_name = "获取资源列表"
    
    try:
        response = requests.get(
            f"{BASE_URL}/resources/",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            resource_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"资源数量: {resource_count}", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_available_resources():
    """测试获取可用资源"""
    test_name = "获取可用资源"
    
    try:
        response = requests.get(
            f"{BASE_URL}/resources/available",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            resource_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"可用资源数量: {resource_count}", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_monitor_health():
    """测试系统健康监控"""
    test_name = "系统健康监控"
    
    try:
        response = requests.get(
            f"{BASE_URL}/monitor/health",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result(test_name, True, 
                        f"CPU: {result.get('cpu_usage')}%, GPU: {result.get('gpu_usage')}%", 
                        result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_get_knowledge_base():
    """测试获取知识库列表"""
    test_name = "获取知识库列表"
    
    try:
        response = requests.get(
            f"{BASE_URL}/knowledge-base/",
            headers=get_headers(),
            params={"skip": 0, "limit": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            kb_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"知识库条目数量: {kb_count}", result[:2] if isinstance(result, list) and kb_count > 0 else result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("用户管理服务测试")
    print("=" * 60)
    print()
    
    # 1. 健康检查
    if not test_health_check():
        print("❌ 服务未启动，请先启动服务: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("开始测试用户管理服务")
    print("-" * 60)
    print()
    
    # 2. 登录获取 Token
    if not test_login():
        print("❌ 登录失败，请检查默认管理员账户是否存在")
        print("   运行: python3 init_db.py 初始化数据库")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("测试认证接口")
    print("-" * 60)
    print()
    
    # 3. 获取当前用户信息
    test_get_current_user()
    
    print()
    print("-" * 60)
    print("测试用户管理接口")
    print("-" * 60)
    print()
    
    # 4. 获取用户列表
    test_get_users()
    
    print()
    print("-" * 60)
    print("测试资源管理接口")
    print("-" * 60)
    print()
    
    # 5. 获取资源列表
    test_get_resources()
    
    # 6. 获取可用资源
    test_get_available_resources()
    
    print()
    print("-" * 60)
    print("测试系统监控接口")
    print("-" * 60)
    print()
    
    # 7. 系统健康监控
    test_get_monitor_health()
    
    print()
    print("-" * 60)
    print("测试知识库接口")
    print("-" * 60)
    print()
    
    # 8. 获取知识库列表
    test_get_knowledge_base()
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
