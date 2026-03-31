import requests
import json
import sys
import os
import datetime

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

BASE_URL = "http://localhost:8000"

def upload_local_file(file_path, user_id="test_user"):
    """模拟前端上传文件"""
    print(f"📤 Uploading file: {file_path}...")
    url = f"{BASE_URL}/api/image/upload" # Correct endpoint for images
    # Or check if there is a generic one. Let's use the one commonly used in other tests usually.
    # Actually, looking at routers, there is /api/{file_category}/upload.
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"user_id": user_id}
            response = requests.post(url, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload Success! URL: {result['url']}")
            return result['url']
        else:
            print(f"❌ Upload Failed: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        return None

def test_local_models():
    # 1. 准备图片
    image_path = os.path.join(TEST_FILES_DIR, "1212.png")
    if not os.path.exists(image_path):
        print(f"❌ Error: Test file not found at {image_path}")
        return

    # 2. 准备图片 Base64 (模拟前端直接传文件)
    print(f"\n[{datetime.datetime.now()}] === Step 1: Preparing Image (Base64) ===")
    import base64
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            base64_image = f"data:image/png;base64,{encoded_string}"
            print(f"✅ Image converted to Base64 (len={len(base64_image)})")
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        return

    # 3. 定义测试任务
    MODELS = [
        {
            "id": "llama3.2:3b",
            "type": "text",
            "params": {"temperature": 0.7, "top_k": 40, "max_tokens": 256, "stream": True},
            "messages": [{"role": "user", "content": "你好，请写一首关于春天的七言绝句"}]
        },
        {
            "id": "llava",
            "type": "vision",
            "params": {"temperature": 0.5, "stream": True},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述这张图片的内容。"},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ]
        },
        {
            "id": "llava-cn",
            "type": "vision",
            "params": {"temperature": 0.5, "stream": True},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述这张图片的内容。"},
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    ]
                }
            ]
        },
        {
            "id": "qwen2.5:7b",
            "type": "text",
            "params": {"temperature": 0.7, "stream": True},
            "messages": [{"role": "user", "content": "你好，请写一首关于春天的七言绝句。"}]
        }
    ]

    # 4. 执行测试
    print(f"\n[{datetime.datetime.now()}] === Step 2: Testing Models ===")
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"🚀 Testing Model: {model['id']} ({model['type']})")
        
        payload = {
            "provider": "local",
            "model": model["id"],
            "messages": model["messages"],
            "config": model["params"]
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/playground/chat", json=payload, stream=True)
            
            full_content = ""
            print(f"[Streaming Output]...")
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            # Handle both OpenAI format and potential raw format
                            content = ""
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                content = data_json["choices"][0].get("delta", {}).get("content", "")
                            elif "content" in data_json:
                                content = data_json["content"]
                                
                            if content:
                                print(content, end="", flush=True)
                                full_content += content
                        except:
                            pass
            
            print(f"\n\n💾 Saving full response...")
            # Save exact artifact
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(TEST_RESULTS_DIR, f"local_{model['id'].replace(':','-')}_{timestamp}.txt")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_content)
                
            print(f"✅ Saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_local_models()
