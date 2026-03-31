"""
报告、知识库与用户文件发送测试
- 报告/知识库：文件上传与数据库记录
- 用户文件：个人发送、批量发送资料/文件给学生
"""
import requests
import json
import sys
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000/api/v1"

# 测试结果输出目录
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# 测试用的 Token 与学生 ID
access_token = None
student_access_token = None
student_user_id = None  # 用于发送文件给学生的 receiver_id

def print_result(test_name, success, message="", data=None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data:
        if isinstance(data, dict):
            # 隐藏敏感信息
            safe_data = data.copy()
            if 'access_token' in safe_data:
                safe_data['access_token'] = safe_data['access_token'][:20] + "..."
            print(f"      响应: {json.dumps(safe_data, ensure_ascii=False, indent=2)}")
        else:
            print(f"      响应: {data}")
    print()

def get_headers(token=None):
    """获取请求头（包含 Token）"""
    headers = {}
    token_to_use = token if token else access_token
    if token_to_use:
        headers["Authorization"] = f"Bearer {token_to_use}"
    return headers

def create_test_file(filename, content=b"test file content"):
    """创建测试文件"""
    test_dir = os.path.join(TEST_RESULTS_DIR, "test_files")
    os.makedirs(test_dir, exist_ok=True)
    
    filepath = os.path.join(test_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(content)
    
    return filepath

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

def test_create_or_login_student():
    """创建或登录学生用户"""
    global student_access_token, student_user_id
    test_name = "创建/登录学生用户"
    
    student_username = "test_student"
    student_password = "test123456"
    student_email = "test_student@test.com"
    
    try:
        # 先尝试登录
        response = requests.post(
            f"{BASE_URL}/auth/login",
            data={
                "username": student_username,
                "password": student_password
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            student_access_token = result.get("access_token")
            student_user_id = result.get("user_id")
            print_result(test_name, True, f"登录成功: {result.get('full_name')}, 角色: {result.get('role')}, ID: {student_user_id}", result)
            return True
        
        # 如果登录失败，尝试创建用户
        print(f"      ⚠️  学生用户不存在，尝试创建...")
        user_data = {
            "username": student_username,
            "email": student_email,
            "password": student_password,
            "full_name": "Test Student",
            "role": "student",
            "student_id": "TEST001"
        }
        
        response = requests.post(
            f"{BASE_URL}/users/",
            json=user_data,
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 201:
            print(f"      ✅ 学生用户创建成功")
            # 创建成功后登录
            response = requests.post(
                f"{BASE_URL}/auth/login",
                data={
                    "username": student_username,
                    "password": student_password
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                student_access_token = result.get("access_token")
                student_user_id = result.get("user_id")
                print_result(test_name, True, f"创建并登录成功: {result.get('full_name')}, 角色: {result.get('role')}, ID: {student_user_id}", result)
                return True
        
        print_result(test_name, False, f"状态码: {response.status_code}", response.json())
        return False
        
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def get_student_headers():
    """获取学生用户的请求头"""
    headers = {}
    if student_access_token:
        headers["Authorization"] = f"Bearer {student_access_token}"
    return headers

def test_submit_report_with_file():
    """测试提交报告（带文件）"""
    test_name = "提交报告（带文件）"
    
    try:
        # 创建测试文件
        test_file_path = create_test_file("test_report.pdf", b"PDF test content for report")
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_report.pdf', f, 'application/pdf')}
            data = {
                'title': '测试报告标题',
                'description': '这是一个测试报告描述'
            }
            
            response = requests.post(
                f"{BASE_URL}/reports/",
                files=files,
                data=data,
                headers=get_student_headers(),  # 使用学生账户
                timeout=30
            )
        
        if response.status_code == 201:
            result = response.json()
            report_id = result.get('id')
            file_id = result.get('file_id')
            
            print_result(test_name, True, 
                        f"报告ID: {report_id}, 文件ID: {file_id}", 
                        result)
            return report_id, file_id
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None, None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None, None

def test_submit_report_without_file():
    """测试提交报告（不带文件）"""
    test_name = "提交报告（不带文件）"
    
    try:
        data = {
            'title': '测试报告标题（无文件）',
            'description': '这是一个不带文件的测试报告描述'
        }
        
        response = requests.post(
            f"{BASE_URL}/reports/",
            data=data,
            headers=get_student_headers(),  # 使用学生账户
            timeout=30
        )
        
        if response.status_code == 201:
            result = response.json()
            report_id = result.get('id')
            file_id = result.get('file_id')
            
            print_result(test_name, True, 
                        f"报告ID: {report_id}, 文件ID: {file_id}", 
                        result)
            return report_id, file_id
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None, None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None, None

def test_get_report(report_id, token=None):
    """测试获取报告详情"""
    test_name = f"获取报告详情 (ID: {report_id})"
    
    try:
        response = requests.get(
            f"{BASE_URL}/reports/{report_id}",
            headers=get_headers(token),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get('file_id')
            print_result(test_name, True, 
                        f"报告标题: {result.get('title')}, 文件ID: {file_id}", 
                        result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_get_my_reports(token=None):
    """测试获取我的报告列表"""
    test_name = "获取我的报告列表"
    
    try:
        response = requests.get(
            f"{BASE_URL}/reports/my-reports",
            headers=get_headers(token),
            params={"skip": 0, "limit": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            report_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, 
                        f"报告数量: {report_count}", 
                        result[:2] if isinstance(result, list) and report_count > 0 else result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_create_knowledge_entry_with_file():
    """测试创建知识库条目（带文件）"""
    test_name = "创建知识库条目（带文件）"
    
    try:
        # 创建测试文件
        test_file_path = create_test_file("test_kb_doc.pdf", b"PDF test content for knowledge base")
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_kb_doc.pdf', f, 'application/pdf')}
            data = {
                'title': '测试知识库条目',
                'content': '这是知识库条目的内容',
                'category': '测试分类',
                'tags': '测试,文档,PDF'
            }
            
            response = requests.post(
                f"{BASE_URL}/knowledge-base/",
                files=files,
                data=data,
                headers=get_headers(),
                timeout=30
            )
        
        if response.status_code == 201:
            result = response.json()
            entry_id = result.get('id')
            file_id = result.get('file_id')
            
            print_result(test_name, True, 
                        f"条目ID: {entry_id}, 文件ID: {file_id}", 
                        result)
            return entry_id, file_id
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None, None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None, None

def test_create_knowledge_entry_without_file():
    """测试创建知识库条目（不带文件）"""
    test_name = "创建知识库条目（不带文件）"
    
    try:
        data = {
            'title': '测试知识库条目（无文件）',
            'content': '这是不带文件的知识库条目内容',
            'category': '测试分类',
            'tags': '测试,文档'
        }
        
        response = requests.post(
            f"{BASE_URL}/knowledge-base/",
            data=data,
            headers=get_headers(),
            timeout=30
        )
        
        if response.status_code == 201:
            result = response.json()
            entry_id = result.get('id')
            file_id = result.get('file_id')
            
            print_result(test_name, True, 
                        f"条目ID: {entry_id}, 文件ID: {file_id}", 
                        result)
            return entry_id, file_id
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None, None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None, None

def test_get_knowledge_entry(entry_id):
    """测试获取知识库条目详情"""
    test_name = f"获取知识库条目详情 (ID: {entry_id})"
    
    try:
        response = requests.get(
            f"{BASE_URL}/knowledge-base/{entry_id}",
            headers=get_headers(),
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get('file_id')
            print_result(test_name, True, 
                        f"条目标题: {result.get('title')}, 文件ID: {file_id}", 
                        result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_get_knowledge_entries():
    """测试获取知识库条目列表"""
    test_name = "获取知识库条目列表"
    
    try:
        response = requests.get(
            f"{BASE_URL}/knowledge-base/",
            headers=get_headers(),
            params={"skip": 0, "limit": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            entry_count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, 
                        f"条目数量: {entry_count}", 
                        result[:2] if isinstance(result, list) and entry_count > 0 else result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_get_user_files():
    """测试获取用户文件列表（验证文件记录是否在数据库中）"""
    test_name = "获取用户文件列表（验证数据库记录）"
    
    try:
        # 测试获取发送的文件
        response = requests.get(
            f"{BASE_URL}/user-files/sent",
            headers=get_headers(),
            params={"skip": 0, "limit": 100},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            file_count = len(result) if isinstance(result, list) else 0
            
            # 检查是否有报告和知识库的文件
            report_files = [f for f in result if isinstance(f, dict) and f.get('file_type') == 'submission']
            kb_files = [f for f in result if isinstance(f, dict) and f.get('file_type') == 'knowledge_base']
            
            print_result(test_name, True, 
                        f"总文件数: {file_count}, 报告文件: {len(report_files)}, 知识库文件: {len(kb_files)}", 
                        {
                            'total': file_count,
                            'report_files': report_files[:3] if report_files else [],
                            'kb_files': kb_files[:3] if kb_files else []
                        })
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None


# ---------- 用户文件发送（个人/批量给学生）----------

def test_send_file_to_student(receiver_id):
    """测试个人发送文件给学生（receiver_id）"""
    test_name = "个人发送文件给学生"
    
    try:
        test_file_path = create_test_file("material_for_student.pdf", b"material content for student")
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('material_for_student.pdf', f, 'application/pdf')}
            data = {'receiver_id': receiver_id}
            
            response = requests.post(
                f"{BASE_URL}/user-files/send",
                files=files,
                data=data,
                headers=get_headers(),
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            rec = result[0] if isinstance(result, list) else result
            fid = rec.get('id')
            rid = rec.get('receiver_id')
            print_result(test_name, True, f"文件ID: {fid}, 接收者ID: {rid}", rec)
            return fid
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_send_file_batch(receiver_ids):
    """测试批量发送文件给学生（batch_data.receiver_ids）"""
    test_name = "批量发送文件给学生"
    
    try:
        test_file_path = create_test_file("material_batch.pdf", b"batch material content")
        batch_data = json.dumps({
            "file_type": "material",
            "description": "批量发放资料",
            "receiver_ids": receiver_ids,
            "is_batch": True
        })
        
        with open(test_file_path, 'rb') as f:
            files = {'file': ('material_batch.pdf', f, 'application/pdf')}
            data = {'batch_data_json': batch_data}
            
            response = requests.post(
                f"{BASE_URL}/user-files/send",
                files=files,
                data=data,
                headers=get_headers(),
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            rec = result[0] if isinstance(result, list) else result
            fid = rec.get('id')
            print_result(test_name, True, f"批量发送成功, 接收者: {receiver_ids}", rec)
            return fid
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_get_received_files(token=None):
    """测试获取我收到的文件（学生视角）"""
    test_name = "获取我收到的文件"
    
    try:
        response = requests.get(
            f"{BASE_URL}/user-files/received",
            headers=get_headers(token),
            params={"skip": 0, "limit": 20},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"收到文件数: {count}", result[:3] if isinstance(result, list) and count else result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_get_sent_files(token=None):
    """测试获取我发出的文件"""
    test_name = "获取我发出的文件"
    
    try:
        response = requests.get(
            f"{BASE_URL}/user-files/sent",
            headers=get_headers(token),
            params={"skip": 0, "limit": 20},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"发出文件数: {count}", result[:3] if isinstance(result, list) and count else result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def main():
    """主测试函数"""
    print("=" * 60)
    print("报告、知识库与用户文件发送测试")
    print("=" * 60)
    print()
    
    # 1. 健康检查
    if not test_health_check():
        print("❌ 服务未启动，请先启动服务: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("开始测试")
    print("-" * 60)
    print()
    
    # 2. 登录获取 Token
    if not test_login():
        print("❌ 登录失败，请检查默认管理员账户是否存在")
        print("   运行: python3 init_db.py 初始化数据库")
        sys.exit(1)
    
    # 2.5. 创建或登录学生用户
    print()
    print("-" * 60)
    print("准备学生账户")
    print("-" * 60)
    print()
    
    if not test_create_or_login_student():
        print("⚠️  无法创建/登录学生用户，报告测试将跳过")
        print("   但知识库测试可以继续（使用admin账户）")
    
    print()
    print("-" * 60)
    print("测试报告文件上传功能")
    print("-" * 60)
    print()
    
    # 3. 测试提交报告（带文件）
    report_id_with_file, file_id_report = test_submit_report_with_file()
    
    # 4. 测试提交报告（不带文件）
    report_id_without_file, _ = test_submit_report_without_file()
    
    # 5. 获取报告详情（使用学生账户）
    if report_id_with_file and student_access_token:
        test_get_report(report_id_with_file, token=student_access_token)
    
    # 6. 获取我的报告列表（使用学生账户）
    if student_access_token:
        test_get_my_reports(token=student_access_token)
    else:
        print("⚠️  跳过获取报告列表（无学生账户）")
    
    print()
    print("-" * 60)
    print("测试知识库文件上传功能")
    print("-" * 60)
    print()
    
    # 7. 测试创建知识库条目（带文件）
    kb_entry_id_with_file, file_id_kb = test_create_knowledge_entry_with_file()
    
    # 8. 测试创建知识库条目（不带文件）
    kb_entry_id_without_file, _ = test_create_knowledge_entry_without_file()
    
    # 9. 获取知识库条目详情
    if kb_entry_id_with_file:
        test_get_knowledge_entry(kb_entry_id_with_file)
    
    # 10. 获取知识库条目列表
    test_get_knowledge_entries()
    
    print()
    print("-" * 60)
    print("验证文件记录是否在数据库中")
    print("-" * 60)
    print()
    
    # 11. 验证文件记录
    test_get_user_files()
    
    print()
    print("-" * 60)
    print("测试用户文件发送（个人/批量给学生）")
    print("-" * 60)
    print()
    
    # 12. 个人发送、批量发送、学生查看收到
    send_individual_ok = False
    send_batch_ok = False
    if student_user_id is not None:
        send_individual_ok = test_send_file_to_student(student_user_id) is not None
        send_batch_ok = test_send_file_batch([student_user_id]) is not None  # 单人也走批量接口
        test_get_sent_files()  # admin 查看发出的文件
        test_get_received_files(token=student_access_token)  # 学生查看收到的文件
    else:
        print("⚠️  无学生账户，跳过用户文件发送测试")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    print()
    print("📋 测试总结:")
    print(f"   - 报告文件上传: {'✅ 成功' if report_id_with_file else '❌ 失败'}")
    print(f"   - 知识库文件上传: {'✅ 成功' if kb_entry_id_with_file else '❌ 失败'}")
    print(f"   - 个人发送文件给学生: {'✅ 成功' if send_individual_ok else '❌ 跳过/失败'}")
    print(f"   - 批量发送文件给学生: {'✅ 成功' if send_batch_ok else '❌ 跳过/失败'}")
    print()
    print("💡 提示:")
    print("   - 如果文件上传成功但数据库中没有记录，请检查:")
    print("     1. 数据库连接是否正常")
    print("     2. 文件上传目录权限是否正确")
    print("     3. 查看服务器日志中的错误信息")
    print("     4. 检查事务是否正确提交")

if __name__ == "__main__":
    main()
