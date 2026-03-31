"""
教师发送给学生的文件 - 测试
- 教师/管理员：上传并发送文件给学生（个人/批量）、查看已发送列表
- 学生：查看收到的文件、查看文件详情
"""
import requests
import json
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000/api/v1"
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

access_token = None
student_access_token = None
student_user_id = None


def print_result(test_name, success, message="", data=None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data is not None and not isinstance(data, list):
        print(f"      响应: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")
    elif isinstance(data, list) and len(data) > 0:
        print(f"      响应: 共 {len(data)} 条, 示例: {json.dumps(data[0], ensure_ascii=False)[:200]}...")
    print()


def get_headers(token=None):
    t = token if token is not None else access_token
    return {"Authorization": f"Bearer {t}"} if t else {}


def create_test_file(filename, content=b"test file content"):
    test_dir = os.path.join(TEST_RESULTS_DIR, "test_files")
    os.makedirs(test_dir, exist_ok=True)
    filepath = os.path.join(test_dir, filename)
    with open(filepath, "wb") as f:
        f.write(content)
    return filepath


def test_health_check():
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        ok = r.status_code == 200
        print_result("健康检查", ok, "服务正常" if ok else f"状态码: {r.status_code}", r.json() if ok else None)
        return ok
    except Exception as e:
        print_result("健康检查", False, str(e))
        return False


def test_login():
    global access_token
    try:
        r = requests.post(
            f"{BASE_URL}/auth/login",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            access_token = data.get("access_token")
            print_result("教师/管理员登录", True, f"用户: {data.get('full_name')}, 角色: {data.get('role')}", data)
            return True
        print_result("教师/管理员登录", False, f"状态码: {r.status_code}", r.json())
        return False
    except Exception as e:
        print_result("教师/管理员登录", False, str(e))
        return False


def test_create_or_login_student():
    global student_access_token, student_user_id
    username, password, email = "test_student", "test123456", "test_student@test.com"
    try:
        r = requests.post(
            f"{BASE_URL}/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            student_access_token = data.get("access_token")
            student_user_id = data.get("user_id")
            print_result("学生登录", True, f"学生: {data.get('full_name')}, ID: {student_user_id}", data)
            return True
        print("      ⚠️  学生用户不存在，尝试创建...")
        r2 = requests.post(
            f"{BASE_URL}/users/",
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": "Test Student",
                "role": "student",
                "student_id": "TEST001",
            },
            headers=get_headers(),
            timeout=10,
        )
        if r2.status_code == 201:
            r3 = requests.post(
                f"{BASE_URL}/auth/login",
                data={"username": username, "password": password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            if r3.status_code == 200:
                data = r3.json()
                student_access_token = data.get("access_token")
                student_user_id = data.get("user_id")
                print_result("学生创建并登录", True, f"学生ID: {student_user_id}", data)
                return True
        print_result("学生创建/登录", False, f"状态码: {r2.status_code}", r2.json() if r2 else None)
        return False
    except Exception as e:
        print_result("学生登录/创建", False, str(e))
        return False


def test_send_file_to_student(receiver_id):
    """教师个人发送文件给学生"""
    test_name = "教师个人发送文件给学生"
    try:
        path = create_test_file("material_for_student.pdf", b"material content for student")
        with open(path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/user-files/send",
                files={"file": ("material_for_student.pdf", f, "application/pdf")},
                data={"receiver_id": str(receiver_id)},  # multipart 下用字符串，后端会解析为 int
                headers=get_headers(),
                timeout=30,
            )
        if r.status_code == 200:
            res = r.json()
            rec = res[0] if isinstance(res, list) else res
            fid = rec.get("id")
            print_result(test_name, True, f"文件ID: {fid}, 接收者ID: {receiver_id}", rec)
            return fid
        print_result(test_name, False, f"状态码: {r.status_code}", r.json())
        return None
    except Exception as e:
        print_result(test_name, False, str(e))
        return None


def test_send_file_batch(receiver_ids):
    """教师批量发送文件给学生"""
    test_name = "教师批量发送文件给学生"
    try:
        path = create_test_file("material_batch.pdf", b"batch material content")
        batch_data = json.dumps({
            "file_type": "material",
            "description": "批量发放资料",
            "receiver_ids": receiver_ids,
            "is_batch": True,
        })
        with open(path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/user-files/send",
                files={"file": ("material_batch.pdf", f, "application/pdf")},
                data={"batch_data_json": batch_data},
                headers=get_headers(),
                timeout=30,
            )
        if r.status_code == 200:
            res = r.json()
            rec = res[0] if isinstance(res, list) else res
            print_result(test_name, True, f"接收者: {receiver_ids}", rec)
            return rec.get("id")
        print_result(test_name, False, f"状态码: {r.status_code}", r.json())
        return None
    except Exception as e:
        print_result(test_name, False, str(e))
        return None


def test_teacher_get_sent_files():
    """教师查看已发送的文件列表"""
    test_name = "教师查看已发送的文件"
    try:
        r = requests.get(
            f"{BASE_URL}/user-files/sent",
            headers=get_headers(),
            params={"skip": 0, "limit": 20},
            timeout=10,
        )
        if r.status_code == 200:
            result = r.json()
            count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"发出文件数: {count}", result[:3] if isinstance(result, list) else result)
            return result
        print_result(test_name, False, f"状态码: {r.status_code}", r.json())
        return None
    except Exception as e:
        print_result(test_name, False, str(e))
        return None


def test_student_get_received_files():
    """学生查看收到的文件列表"""
    test_name = "学生查看收到的文件"
    try:
        r = requests.get(
            f"{BASE_URL}/user-files/received",
            headers=get_headers(student_access_token),
            params={"skip": 0, "limit": 20},
            timeout=10,
        )
        if r.status_code == 200:
            result = r.json()
            count = len(result) if isinstance(result, list) else 0
            print_result(test_name, True, f"收到文件数: {count}", result[:3] if isinstance(result, list) else result)
            return result
        print_result(test_name, False, f"状态码: {r.status_code}", r.json())
        return None
    except Exception as e:
        print_result(test_name, False, str(e))
        return None


def test_student_get_file_detail(file_id):
    """学生查看某个收到文件的详情"""
    test_name = "学生查看文件详情"
    try:
        r = requests.get(
            f"{BASE_URL}/user-files/{file_id}",
            headers=get_headers(student_access_token),
            timeout=10,
        )
        if r.status_code == 200:
            result = r.json()
            print_result(test_name, True, f"文件ID: {file_id}", result)
            return result
        print_result(test_name, False, f"状态码: {r.status_code}", r.json())
        return None
    except Exception as e:
        print_result(test_name, False, str(e))
        return None


def main():
    print("=" * 60)
    print("教师发送给学生的文件 - 测试")
    print("=" * 60)
    print()

    if not test_health_check():
        print("❌ 服务未启动，请先: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    if not test_login():
        print("❌ 登录失败，请检查默认管理员账户或运行 init_db.py")
        sys.exit(1)

    if not test_create_or_login_student():
        print("❌ 无学生账户，无法继续测试「学生查看收到文件」")
        sys.exit(1)

    print("-" * 60)
    print("教师发文件 + 学生查看")
    print("-" * 60)
    print()

    test_send_file_to_student(student_user_id)
    test_send_file_batch([student_user_id])
    test_teacher_get_sent_files()
    received = test_student_get_received_files()

    if isinstance(received, list) and len(received) > 0:
        first_id = received[0].get("id") if isinstance(received[0], dict) else None
        if first_id:
            test_student_get_file_detail(first_id)

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
