"""
文件上传功能测试
测试视频、音频、图片上传接口

教师发文件给学生、学生查看收到文件：见 tests/test_teacher_student_files.py
"""
import requests
import json
import sys
import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "http://localhost:8000"

# 测试结果输出目录
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

def print_result(test_name, success, message="", data=None):
    """打印测试结果"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} | {test_name}")
    if message:
        print(f"      {message}")
    if data:
        print(f"      响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
    print()

def create_test_file(file_type, content=b"test content"):
    """创建测试文件"""
    test_dir = os.path.join(TEST_RESULTS_DIR, "test_files")
    os.makedirs(test_dir, exist_ok=True)
    
    extensions = {
        'video': 'mp4',
        'audio': 'mp3',
        'image': 'jpg'
    }
    
    filename = f"test_{file_type}.{extensions[file_type]}"
    filepath = os.path.join(test_dir, filename)
    
    with open(filepath, 'wb') as f:
        f.write(content)
    
    return filepath, filename

def test_upload(file_category, file_path):
    """测试文件上传"""
    test_name = f"上传{file_category}"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, f'application/{file_category}')}
            data = {'user_id': 'test_user_123'}
            
            response = requests.post(
                f"{BASE_URL}/api/{file_category}/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            file_id = result.get(f"{file_category}_id")
            url = result.get("url")
            
            print_result(test_name, True, f"文件ID: {file_id}", result)
            return file_id, url
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None, None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None, None

def test_get_file(file_category, file_id):
    """测试获取文件URL"""
    test_name = f"获取{file_category} URL"
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/{file_category}/{file_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            url = result.get("url")
            print_result(test_name, True, f"URL: {url}", result)
            return url
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_list_files(file_category, user_id=None):
    """测试获取文件列表"""
    test_name = f"获取{file_category}列表"
    
    try:
        params = {'page': 1, 'per_page': 10}
        if user_id:
            params['user_id'] = user_id
        
        response = requests.get(
            f"{BASE_URL}/api/{file_category}/list",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            files = result.get(f"{file_category}s", [])
            pagination = result.get("pagination", {})
            print_result(test_name, True, 
                        f"总数: {pagination.get('total', 0)}, 当前页: {len(files)}", 
                        result)
            return result
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return None
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return None

def test_delete_file(file_category, file_id):
    """测试删除文件"""
    test_name = f"删除{file_category}"
    
    try:
        response = requests.delete(
            f"{BASE_URL}/api/{file_category}/{file_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print_result(test_name, True, "删除成功", result)
            return True
        else:
            print_result(test_name, False, f"状态码: {response.status_code}", response.json())
            return False
    except Exception as e:
        print_result(test_name, False, f"错误: {str(e)}")
        return False

def test_health_check():
    """测试健康检查"""
    test_name = "健康检查"
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
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

def main():
    """主测试函数"""
    print("=" * 60)
    print("文件上传功能测试")
    print("=" * 60)
    print()
    
    # 1. 健康检查
    if not test_health_check():
        print("❌ 服务未启动，请先启动服务: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    print()
    print("-" * 60)
    print("开始测试文件上传功能")
    print("-" * 60)
    print()
    
    # 测试三种文件类型
    file_categories = ['video', 'audio', 'image']
    uploaded_files = {}
    
    for file_category in file_categories:
        print(f"\n📁 测试 {file_category.upper()} 上传功能")
        print("-" * 60)
        
        file_path = None
        
        # 特别处理图片：尝试使用用户提供的 1212.png
        if file_category == 'image':
            possible_paths = [
                os.path.join(PROJECT_ROOT, "测试文件", "1212.png"),
                os.path.join(PROJECT_ROOT, "1212.png")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"ℹ️  发现用户提供的测试图片: {path}")
                    file_path, filename = path, os.path.basename(path)
                    break
            
            if not file_path:
                 print(f"ℹ️  未找到 1212.png (搜索路径: {possible_paths})，将生成测试图片")
        
        # 如果没有找到用户文件，则创建测试文件
        if not file_path:
            file_path, filename = create_test_file(file_category, b"test file content for " + file_category.encode())
        
        # 测试上传
        file_id, url = test_upload(file_category, file_path)
        if file_id:
            uploaded_files[file_category] = file_id
            
            # 测试获取URL
            test_get_file(file_category, file_id)
            
            # 测试列表
            test_list_files(file_category, 'test_user_123')
            
            # 测试删除
            test_delete_file(file_category, file_id)
        else:
            print(f"⚠️  {file_category} 上传失败，跳过后续测试")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 清理测试文件
    test_files_dir = os.path.join(TEST_RESULTS_DIR, "test_files")
    if os.path.exists(test_files_dir):
        import shutil
        shutil.rmtree(test_files_dir)
        print(f"✅ 已清理测试文件: {test_files_dir}")

if __name__ == "__main__":
    main()